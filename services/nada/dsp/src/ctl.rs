//! The control thread. It owns every profile, impulse response and insert record, builds
//! stages off the audio path and publishes them. Binder threads and audio threads only post
//! messages to it.

use crate::config::{self, Plan, Settings};
use crate::conv::{Convolver, Kernel};
use crate::engine::{self, Shared, STAGES, STAGE_NAMES};
use crate::ir::{resample, Store};
use crate::slot::StageBox;
use crate::colour::Colour;
use crate::dyneq::DynEq;
use crate::lookahead::LookLimiter;
use crate::loud::Loud;
use crate::mbc::Mbc;
use crate::stages::{Bass, Comp, Eq, Limiter, Spatial, Trim};
use crate::Stage;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering::Relaxed};
use std::sync::mpsc::{channel, sync_channel, Sender, SyncSender};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

pub const ROUTES: [&str; 6] = ["none", "spk", "wired", "usb", "bt", "other"];

pub enum Msg {
    Params(String),
    Add(Arc<Shared>),
    Remove(u64),
    Rate(u64),
    Route(i32, u8),
    Query(String, SyncSender<String>),
}

const DENY: usize = 64;
static MMAP_DENY: [AtomicU32; DENY] = [const { AtomicU32::new(0) }; DENY];

/// Whether this uid asked for processing and so must not get an MMAP stream. Lock-free.
pub fn avoid_mmap(uid: u32) -> bool {
    uid != 0 && MMAP_DENY.iter().any(|d| d.load(Relaxed) == uid)
}

static PROCESSING: AtomicBool = AtomicBool::new(false);

/// Whether any device class has a profile with an active stage. Lock-free; audio policy
/// reads it to keep media off compress offload, which no mixer ever sees.
pub fn processing() -> bool {
    PROCESSING.load(Relaxed)
}

static TX: OnceLock<Mutex<Sender<Msg>>> = OnceLock::new();

pub fn post(m: Msg) {
    let tx = TX.get_or_init(|| {
        let (tx, rx) = channel::<Msg>();
        let _ = std::thread::Builder::new().name("nada-ctl".into()).spawn(move || {
            let mut c = Ctl::default();
            for m in rx {
                c.handle(m);
            }
        });
        Mutex::new(tx)
    });
    if let Ok(tx) = tx.lock() {
        let _ = tx.send(m);
    }
}

/// Asks the control thread and waits briefly for the answer.
pub fn query(key: &str) -> String {
    let (tx, rx) = sync_channel(1);
    post(Msg::Query(key.to_string(), tx));
    rx.recv_timeout(Duration::from_millis(1000)).unwrap_or_default()
}

struct Ins {
    sh: Arc<Shared>,
    applied: Plan,
    rate: u32,
}

#[derive(Default)]
pub struct Ctl {
    ins: Vec<Ins>,
    routes: HashMap<i32, u8>,
    cfg: [Option<Settings>; 6],
    irs: Store,
    kernels: HashMap<(String, u64, u32), Arc<Kernel>>,
    last_error: String,
}

fn value<'a>(kv: &'a str, key: &str) -> Option<&'a str> {
    kv.split(';').find_map(|p| p.strip_prefix(key)?.strip_prefix('='))
}

impl Ctl {
    pub fn handle(&mut self, m: Msg) {
        match m {
            Msg::Params(kv) => {
                if let Err(e) = self.params(&kv) {
                    self.last_error = e;
                }
            }
            Msg::Add(sh) => {
                self.ins.push(Ins { sh, applied: Plan::default(), rate: 0 });
                self.reconcile();
            }
            Msg::Remove(id) => self.ins.retain(|i| i.sh.id != id),
            Msg::Rate(id) => {
                for i in self.ins.iter_mut().filter(|i| i.sh.id == id) {
                    i.rate = 0;
                }
                self.reconcile();
            }
            Msg::Route(io, r) => {
                self.routes.insert(io, r.min(5));
                self.reconcile();
            }
            Msg::Query(key, tx) => {
                let _ = tx.send(self.query(&key));
            }
        }
    }

    fn params(&mut self, kv: &str) -> Result<(), String> {
        if let Some(v) = value(kv, "nada_ir") {
            self.irs.begin(v)?;
        }
        if let Some(v) = value(kv, "nada_ird") {
            self.irs.data(v)?;
        }
        if let Some(v) = value(kv, "nada_ire") {
            self.irs.commit(v)?;
            self.kernels.retain(|k, _| k.0 != v);
        }
        if let Some(v) = value(kv, "nada_irx") {
            self.irs.irs.remove(v);
            self.kernels.retain(|k, _| k.0 != v);
        }
        if let Some(v) = value(kv, "nada_cfg") {
            let (route, spec) = v.split_once('|').unwrap_or((v, ""));
            let s = Settings::parse(spec)?;
            match ROUTES.iter().position(|r| *r == route) {
                Some(0) | None if route != "all" => return Err(format!("bad route '{route}'")),
                Some(r) => self.cfg[r] = Some(s),
                None => self.cfg[1..].iter_mut().for_each(|c| *c = Some(s.clone())),
            }
        }
        if let Some(v) = value(kv, "nada_mmap") {
            let uids: Vec<u32> = v.split(',').filter_map(|u| u.trim().parse().ok()).take(DENY).collect();
            for (i, d) in MMAP_DENY.iter().enumerate() {
                d.store(uids.get(i).copied().unwrap_or(0), Relaxed);
            }
        }
        if value(kv, "nada_diag") == Some("reset") {
            for i in &self.ins {
                i.sh.tele.clear();
                i.sh.lim.blocks.store(0, Relaxed);
                i.sh.lim.max_cdb.store(0, Relaxed);
            }
        }
        let mut none = |_: &str| Some((0u64, 1.0));
        let on = self.cfg.iter().flatten().any(|c| config::plan(c, 48000.0, true, &mut none).active());
        PROCESSING.store(on, Relaxed);
        self.reconcile();
        Ok(())
    }

    fn kernel(&mut self, id: &str, rate: u32) -> Option<Arc<Kernel>> {
        let ir = self.irs.irs.get(id)?.clone();
        let key = (id.to_string(), ir.gen, rate);
        if let Some(k) = self.kernels.get(&key) {
            return Some(k.clone());
        }
        let ch: Vec<Vec<f32>> = ir.ch.iter().map(|c| resample(c, ir.rate, rate)).collect();
        let k = Arc::new(Kernel::new(&ch)?);
        self.kernels.insert(key, k.clone());
        Some(k)
    }

    // Brings every insert to what its route's profile asks for, touching only the stages
    // whose parameters changed.
    fn reconcile(&mut self) {
        for n in 0..self.ins.len() {
            let sh = self.ins[n].sh.clone();
            let rate = sh.rate.load(Relaxed);
            if rate == 0 {
                continue;
            }
            let route = self.routes.get(&sh.io).copied().unwrap_or(0) as usize;
            let settings = self.cfg[route].clone().unwrap_or_default();
            let fs = rate as f64;
            let mut gains: HashMap<String, Option<(u64, f64)>> = HashMap::new();
            if !settings.ir.is_empty() {
                let g = self.kernel(&settings.ir, rate).zip(self.irs.irs.get(&settings.ir)).map(|(k, ir)| (ir.gen, k.peak_gain));
                gains.insert(settings.ir.clone(), g);
            }
            let plan = config::plan(&settings, fs, sh.may_delay(), &mut |id| gains.get(id).copied().flatten());
            let fresh = self.ins[n].rate != rate;
            let old = if fresh { Plan::default() } else { self.ins[n].applied.clone() };
            let mut changed = fresh;
            for st in 0..STAGES {
                let same = match st {
                    engine::TRIM => plan.trim == old.trim,
                    engine::CONV => plan.conv == old.conv,
                    engine::EQ => plan.eq == old.eq,
                    engine::DYNEQ => plan.dyneq == old.dyneq,
                    engine::BASS => plan.bass == old.bass,
                    engine::COLOUR => plan.colour == old.colour,
                    engine::SPATIAL => plan.spatial == old.spatial,
                    engine::MBC => plan.mbc == old.mbc,
                    engine::COMP => plan.comp == old.comp,
                    engine::LOUD => plan.loud == old.loud,
                    _ => plan.lim == old.lim,
                };
                if same && !fresh {
                    continue;
                }
                let stage: Option<Box<dyn Stage>> = match st {
                    engine::TRIM => plan.trim.map(|g| Box::new(Trim(g)) as _),
                    engine::CONV => plan.conv.as_ref().and_then(|c| {
                        let k = self.kernel(&c.ir, rate)?;
                        Some(Box::new(Convolver::new(k, c.wet, c.dry)) as _)
                    }),
                    engine::EQ => (!plan.eq.is_empty()).then(|| Box::new(Eq::new(&plan.eq)) as _),
                    engine::DYNEQ => (!plan.dyneq.is_empty()).then(|| Box::new(DynEq::new(&plan.dyneq, fs)) as _),
                    engine::BASS => plan.bass.as_ref().map(|p| Box::new(Bass::new(p, fs)) as _),
                    engine::COLOUR => plan.colour.as_ref().map(|p| Box::new(Colour::new(p, fs)) as _),
                    engine::SPATIAL => plan.spatial.as_ref().map(|p| Box::new(Spatial::new(p, fs)) as _),
                    engine::MBC => plan.mbc.as_ref().map(|p| Box::new(Mbc::new(p, fs)) as _),
                    engine::COMP => plan.comp.as_ref().map(|p| Box::new(Comp::new(p, fs)) as _),
                    engine::LOUD => plan.loud.as_ref().map(|p| Box::new(Loud::new(p, fs)) as _),
                    _ => plan.lim.as_ref().map(|p| -> Box<dyn Stage> {
                        if p.lookahead > 0 {
                            Box::new(LookLimiter::new(p, fs, sh.lim.clone()))
                        } else {
                            Box::new(Limiter::new(p, fs, sh.lim.clone()))
                        }
                    }),
                };
                if fresh && stage.is_none() {
                    continue;
                }
                sh.slots[st].publish(StageBox { rate, stage });
                changed = true;
            }
            sh.latency.store(plan.latency, Relaxed);
            if changed {
                sh.tele.overload.store(false, Relaxed);
                sh.wake();
            }
            self.ins[n].applied = plan;
            self.ins[n].rate = rate;
            sh.slots.iter().for_each(|s| s.drain());
        }
    }

    fn query(&mut self, key: &str) -> String {
        match key {
            "nada_schema" => config::schema_text(),
            "nada_active" => (processing() as u8).to_string(),
            "nada_dump" => self.dump(),
            _ => String::new(),
        }
    }

    fn dump(&self) -> String {
        let mut s = format!("nada schema={} inserts={}\n", config::SCHEMA_VERSION, self.ins.len());
        for i in &self.ins {
            let (sh, t) = (&i.sh, &i.sh.tele);
            let route = self.routes.get(&sh.io).copied().unwrap_or(0) as usize;
            let p = &i.applied;
            let on = [
                p.trim.is_some(),
                p.conv.is_some(),
                !p.eq.is_empty(),
                !p.dyneq.is_empty(),
                p.bass.is_some(),
                p.colour.is_some(),
                p.spatial.is_some(),
                p.mbc.is_some(),
                p.comp.is_some(),
                p.loud.is_some(),
                p.lim.is_some(),
            ];
            let stages: Vec<&str> = STAGE_NAMES.iter().zip(on).filter(|(_, o)| *o).map(|(n, _)| *n).collect();
            let calls = t.calls.load(Relaxed);
            s += &format!(
                "insert io={} kind={} rate={} route={} stages={} pre_db={:.2} boost_db={:.2} added_latency_frames={} skipped={} \
                 calls={} frames={} avg_ns={} p50_ns={} p99_ns={} max_ns={} late={} nonfinite={} overload={} \
                 lim_blocks={} lim_max_db={:.2}\n",
                sh.io,
                if sh.kind == 0 { "fast" } else { "mixer" },
                sh.rate.load(Relaxed),
                ROUTES[route],
                if stages.is_empty() { "-".into() } else { stages.join(",") },
                p.pre_db,
                p.boost_db,
                p.latency,
                if p.skipped.is_empty() { "-".into() } else { p.skipped.join(",") },
                calls,
                t.frames.load(Relaxed),
                if calls > 0 { t.ns_sum.load(Relaxed) / calls } else { 0 },
                t.quantile(0.5),
                t.quantile(0.99),
                t.ns_max.load(Relaxed),
                t.late.load(Relaxed),
                t.nonfinite.load(Relaxed),
                t.overload.load(Relaxed) as u8,
                sh.lim.blocks.load(Relaxed),
                sh.lim.max_cdb.load(Relaxed) as f64 / 100.0,
            );
        }
        for (r, c) in self.cfg.iter().enumerate() {
            if let Some(c) = c {
                s += &format!("cfg {} {}\n", ROUTES[r], c.spec());
            }
        }
        for (id, ir) in &self.irs.irs {
            s += &format!("ir {} rate={} channels={} frames={}\n", id, ir.rate, ir.ch.len(), ir.ch[0].len());
        }
        let deny: Vec<String> = MMAP_DENY.iter().map(|d| d.load(Relaxed)).filter(|u| *u != 0).map(|u| u.to_string()).collect();
        s += &format!("mmap_deny {}\n", deny.join(","));
        if !self.last_error.is_empty() {
            s += &format!("last_error {}\n", self.last_error);
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Engine;

    fn rig(io: i32) -> (Ctl, Engine) {
        let sh = Arc::new(Shared::new(io as u64, io, 0));
        let mut e = Engine::new(sh.clone());
        e.set_rate(48000);
        let mut c = Ctl::default();
        c.handle(Msg::Add(sh));
        c.handle(Msg::Route(io, 1));
        (c, e)
    }

    fn tone(e: &mut Engine, f: f64, calls: usize) -> f64 {
        let mut n = 0usize;
        let mut peak = 0f32;
        for _ in 0..calls {
            let mut b: Vec<f32> = (0..144)
                .flat_map(|i| {
                    let v = (0.1 * (2.0 * std::f64::consts::PI * f * (n + i) as f64 / 48000.0).sin()) as f32;
                    [v, v]
                })
                .collect();
            n += 144;
            e.process(&mut b, 2);
            peak = b.iter().fold(0f32, |a, v| a.max(v.abs()));
        }
        20.0 * (peak as f64 / 0.1).log10()
    }

    #[test]
    fn profile_reaches_the_audio_thread_by_route() {
        let (mut c, mut e) = rig(13);
        assert!(tone(&mut e, 1000.0, 20).abs() < 1e-6);
        c.handle(Msg::Params("nada_cfg=spk|on:1|pre.auto:0|eq.on:1|eq.bands:pk/1000/-6/1.4".into()));
        assert!((tone(&mut e, 1000.0, 400) + 6.0).abs() < 0.05);
        // Another route's profile does not touch this insert.
        c.handle(Msg::Params("nada_cfg=bt|on:1|pre.auto:0|eq.on:1|eq.bands:pk/1000/-12/1.4".into()));
        assert!((tone(&mut e, 1000.0, 400) + 6.0).abs() < 0.05);
        c.handle(Msg::Route(13, 4));
        assert!((tone(&mut e, 1000.0, 400) + 12.0).abs() < 0.05);
        // The earpiece is never processed.
        c.handle(Msg::Route(13, 0));
        assert!(tone(&mut e, 1000.0, 400).abs() < 1e-6);
        assert!(c.dump().contains("route=none stages=-"));
    }

    #[test]
    fn unchanged_stages_are_not_rebuilt() {
        let (mut c, mut e) = rig(21);
        c.handle(Msg::Params("nada_cfg=all|on:1|eq.on:1|eq.bands:pk/100/3/1|bass.on:1".into()));
        tone(&mut e, 1000.0, 50);
        // Only the bass stage changes: the equaliser's slot gets nothing new.
        c.handle(Msg::Params("nada_cfg=all|on:1|eq.on:1|eq.bands:pk/100/3/1|bass.on:1|bass.gain:6".into()));
        assert!(!e.shared.slots[engine::EQ].has_pending());
        assert!(e.shared.slots[engine::BASS].has_pending());
        c.handle(Msg::Params("nada_cfg=all|nope:1".into()));
        assert!(c.dump().contains("last_error unknown key"));
    }

    #[test]
    fn convolver_from_upload() {
        let (mut c, mut e) = rig(5);
        let ir = [0.5f32, 0.0, 0.0, 0.0];
        let mut crc = crate::ir::Crc::new();
        ir.iter().for_each(|v| crc.feed(&v.to_le_bytes()));
        c.handle(Msg::Params(format!("nada_ir=half,48000,1,4,{:08x}", crc.done())));
        c.handle(Msg::Params("nada_ird=half,0,AAAAPwAAAAAAAAAAAAAAAA==".into()));
        c.handle(Msg::Params("nada_ire=half".into()));
        c.handle(Msg::Params("nada_cfg=spk|on:1|conv.on:1|conv.ir:half".into()));
        assert!((tone(&mut e, 1000.0, 400) + 6.02).abs() < 0.01);
        assert!(c.dump().contains("stages=conv,lim"));
    }

    #[test]
    fn mmap_deny_list() {
        let mut c = Ctl::default();
        c.handle(Msg::Params("nada_mmap=10123,10456".into()));
        assert!(avoid_mmap(10456) && !avoid_mmap(10999) && !avoid_mmap(0));
        c.handle(Msg::Params("nada_mmap=".into()));
        assert!(!avoid_mmap(10456));
    }
}
