//! The settings schema and the step from settings to stage parameters.
//!
//! `SCHEMA` is the single definition of every setting: key, range, step, default, unit and
//! title. The service validates against it and the settings pages are drawn from it, so a new
//! setting is one row here plus the code in `plan` that reads it.

use crate::biquad::{design, Coef, Kind};
use crate::colour::{self, ColourP};
use crate::dyneq::DynBand;
use crate::lookahead::MIN_LOOKAHEAD;
use crate::loud::LoudP;
use crate::mbc::MbcP;
use crate::stages::{BassP, CompP, LimP, SpatialP};
use crate::{db, to_db};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Ty {
    Bool,
    Num,
    /// Equaliser band list, `type/freq/gain/q` separated by commas.
    Bands,
    /// Name of an uploaded impulse response.
    Text,
}

pub struct Param {
    pub key: &'static str,
    pub group: &'static str,
    pub ty: Ty,
    pub min: f64,
    pub max: f64,
    pub step: f64,
    pub def: f64,
    pub unit: &'static str,
    pub title: &'static str,
}

const fn b(key: &'static str, group: &'static str, def: bool, title: &'static str) -> Param {
    Param { key, group, ty: Ty::Bool, min: 0.0, max: 1.0, step: 1.0, def: if def { 1.0 } else { 0.0 }, unit: "", title }
}

#[allow(clippy::too_many_arguments)]
const fn n(
    key: &'static str,
    group: &'static str,
    min: f64,
    max: f64,
    step: f64,
    def: f64,
    unit: &'static str,
    title: &'static str,
) -> Param {
    Param { key, group, ty: Ty::Num, min, max, step, def, unit, title }
}

const fn t(key: &'static str, group: &'static str, ty: Ty, title: &'static str) -> Param {
    Param { key, group, ty, min: 0.0, max: 0.0, step: 0.0, def: 0.0, unit: "", title }
}

pub const SCHEMA_VERSION: u32 = 4;

/// Delay a mix point that may hold audio back gives the chain, in ms. Constant while the
/// chain is active, so stages turning on and off never move the timestamps.
pub const DELAY_BUDGET_MS: f64 = 2.0;

/// One effect on the settings pages: a switch named `<id>.on` (none for `main` and `out`),
/// the controls shown while it is on, a kind the pages show as a badge, and one line that
/// says what it does to the sound.
pub struct Group {
    pub id: &'static str,
    pub title: &'static str,
    /// `master`, `tone` (linear), `dynamics`, `colour` (adds harmonics), `spatial`, `output`.
    pub kind: &'static str,
    pub summary: &'static str,
}

const fn g(id: &'static str, title: &'static str, kind: &'static str, summary: &'static str) -> Group {
    Group { id, title, kind, summary }
}

/// In display order, which is also the order of the chain.
pub const GROUPS: &[Group] = &[
    g("main", "Processor effects", "master", "Runs on the phone's processor, after Dolby's controls and before the DSP chip. Automatic headroom lowers the input by the largest boost below, so nothing clips."),
    g("conv", "Convolver", "tone", "Applies an impulse response: a room, a speaker or headphone correction. Exact, no added delay; uses the most processor time."),
    g("eq", "Equaliser", "tone", "Linear tone control, no added delay. Transparent when flat. Cuts are safer than boosts."),
    g("dyneq", "Dynamic equaliser", "dynamics", "A band moves only while it is loud: tames harsh highs or boomy bass in loud passages and leaves quiet ones alone."),
    g("bass", "Bass", "tone", "A low shelf boost, and psychoacoustic bass for small speakers: overtones of the bass the speaker cannot play, so the ear still hears it. With a speaker limit set, the unplayable fundamental is removed. The overtones add colour."),
    g("exc", "Exciter", "colour", "Adds harmonics above the corner: brighter and more present, not more accurate. Music output only; games are not delayed by it."),
    g("tube", "Tube", "colour", "Asymmetric saturation like a triode, at 4x the sample rate so nothing folds back. Warmth, not fidelity. Music output only."),
    g("xfeed", "Crossfeed", "spatial", "For headphones: blends a little of each side into the other, as speakers in a room do. Less fatigue on hard-panned recordings."),
    g("width", "Stereo width", "spatial", "Mid/side balance in three bands: above 100 % wider, below narrower, 0 % mono. Keeping the bass at 100 % or less avoids a hollow low end."),
    g("mbc", "Multiband compressor", "dynamics", "Three bands, each compressed only when it alone is loud: loud bass no longer pushes the voice down. Louder small speakers, less dynamic range. The crossovers turn the phase even when nothing is compressed."),
    g("comp", "Compressor", "dynamics", "Narrows the loudness range. Leave off for the most faithful sound; useful at night or in noise."),
    g("loud", "Loudness levelling", "dynamics", "Slowly brings every app and track to one loudness (BS.1770 meter, LUFS). Holds during silence, so noise is never pulled up."),
    g("out", "Output", "output", "Output gain, then a limiter that leaves the signal untouched below the ceiling. On music output it reads peaks between samples too."),
];

pub const SCHEMA: &[Param] = &[
    b("on", "main", false, "Processor effects"),
    b("pre.auto", "main", true, "Automatic headroom"),
    n("pre.gain", "main", -24.0, 0.0, 0.1, 0.0, "dB", "Preamp (when automatic is off)"),
    b("conv.on", "conv", false, "Convolver"),
    t("conv.ir", "conv", Ty::Text, "Impulse response"),
    n("conv.wet", "conv", -24.0, 12.0, 0.5, 0.0, "dB", "Level"),
    n("conv.dry", "conv", 0.0, 100.0, 1.0, 0.0, "%", "Dry mix"),
    b("eq.on", "eq", false, "Equaliser"),
    t("eq.bands", "eq", Ty::Bands, "Bands"),
    b("dyneq.on", "dyneq", false, "Dynamic equaliser"),
    n("dyneq.1.freq", "dyneq", 20.0, 20000.0, 10.0, 150.0, "Hz", "Band 1"),
    n("dyneq.1.q", "dyneq", 0.3, 4.0, 0.1, 1.0, "", "Band 1 Q"),
    n("dyneq.1.threshold", "dyneq", -60.0, 0.0, 1.0, -24.0, "dB", "Band 1 threshold"),
    n("dyneq.1.range", "dyneq", -12.0, 12.0, 0.5, -3.0, "dB", "Band 1 range (0 is off)"),
    n("dyneq.1.attack", "dyneq", 0.5, 100.0, 0.5, 5.0, "ms", "Band 1 attack"),
    n("dyneq.1.release", "dyneq", 10.0, 1000.0, 5.0, 120.0, "ms", "Band 1 release"),
    n("dyneq.2.freq", "dyneq", 20.0, 20000.0, 10.0, 3500.0, "Hz", "Band 2"),
    n("dyneq.2.q", "dyneq", 0.3, 4.0, 0.1, 1.5, "", "Band 2 Q"),
    n("dyneq.2.threshold", "dyneq", -60.0, 0.0, 1.0, -30.0, "dB", "Band 2 threshold"),
    n("dyneq.2.range", "dyneq", -12.0, 12.0, 0.5, -4.0, "dB", "Band 2 range (0 is off)"),
    n("dyneq.2.attack", "dyneq", 0.5, 100.0, 0.5, 5.0, "ms", "Band 2 attack"),
    n("dyneq.2.release", "dyneq", 10.0, 1000.0, 5.0, 120.0, "ms", "Band 2 release"),
    n("dyneq.3.freq", "dyneq", 20.0, 20000.0, 10.0, 60.0, "Hz", "Band 3"),
    n("dyneq.3.q", "dyneq", 0.3, 4.0, 0.1, 1.0, "", "Band 3 Q"),
    n("dyneq.3.threshold", "dyneq", -60.0, 0.0, 1.0, -40.0, "dB", "Band 3 threshold"),
    n("dyneq.3.range", "dyneq", -12.0, 12.0, 0.5, 0.0, "dB", "Band 3 range (0 is off)"),
    n("dyneq.3.attack", "dyneq", 0.5, 100.0, 0.5, 5.0, "ms", "Band 3 attack"),
    n("dyneq.3.release", "dyneq", 10.0, 1000.0, 5.0, 120.0, "ms", "Band 3 release"),
    n("dyneq.4.freq", "dyneq", 20.0, 20000.0, 10.0, 8000.0, "Hz", "Band 4"),
    n("dyneq.4.q", "dyneq", 0.3, 4.0, 0.1, 2.0, "", "Band 4 Q"),
    n("dyneq.4.threshold", "dyneq", -60.0, 0.0, 1.0, -36.0, "dB", "Band 4 threshold"),
    n("dyneq.4.range", "dyneq", -12.0, 12.0, 0.5, 0.0, "dB", "Band 4 range (0 is off)"),
    n("dyneq.4.attack", "dyneq", 0.5, 100.0, 0.5, 5.0, "ms", "Band 4 attack"),
    n("dyneq.4.release", "dyneq", 10.0, 1000.0, 5.0, 120.0, "ms", "Band 4 release"),
    b("bass.on", "bass", false, "Bass"),
    n("bass.gain", "bass", 0.0, 12.0, 0.5, 4.0, "dB", "Boost"),
    n("bass.freq", "bass", 40.0, 200.0, 5.0, 100.0, "Hz", "Corner"),
    n("bass.harm", "bass", 0.0, 100.0, 1.0, 0.0, "%", "Psychoacoustic bass"),
    n("bass.cut", "bass", 0.0, 300.0, 5.0, 0.0, "Hz", "Speaker low limit (0 keeps the fundamental)"),
    b("exc.on", "exc", false, "Exciter"),
    n("exc.amount", "exc", 0.0, 100.0, 1.0, 30.0, "%", "Amount"),
    n("exc.freq", "exc", 1000.0, 10000.0, 100.0, 3000.0, "Hz", "Corner"),
    b("tube.on", "tube", false, "Tube"),
    n("tube.drive", "tube", 0.0, 100.0, 1.0, 30.0, "%", "Drive"),
    b("xfeed.on", "xfeed", false, "Crossfeed"),
    n("xfeed.level", "xfeed", 1.0, 12.0, 0.5, 4.5, "dB", "Level"),
    n("xfeed.freq", "xfeed", 400.0, 1200.0, 50.0, 700.0, "Hz", "Corner"),
    b("width.on", "width", false, "Stereo width"),
    n("width.amount", "width", 0.0, 200.0, 5.0, 120.0, "%", "Width, middle band"),
    n("width.low", "width", 0.0, 200.0, 5.0, 100.0, "%", "Width, bass"),
    n("width.high", "width", 0.0, 200.0, 5.0, 120.0, "%", "Width, treble"),
    n("width.xlow", "width", 60.0, 500.0, 10.0, 150.0, "Hz", "Bass up to"),
    n("width.xhigh", "width", 1000.0, 12000.0, 100.0, 4000.0, "Hz", "Treble from"),
    b("mbc.on", "mbc", false, "Multiband compressor"),
    n("mbc.xlow", "mbc", 60.0, 1000.0, 10.0, 200.0, "Hz", "Low band up to"),
    n("mbc.xhigh", "mbc", 1000.0, 10000.0, 100.0, 4000.0, "Hz", "High band from"),
    n("mbc.threshold", "mbc", -50.0, 0.0, 0.5, -24.0, "dB", "Threshold"),
    n("mbc.ratio", "mbc", 1.0, 10.0, 0.1, 2.5, ":1", "Ratio"),
    n("mbc.attack", "mbc", 0.5, 100.0, 0.5, 10.0, "ms", "Attack (middle band)"),
    n("mbc.release", "mbc", 20.0, 1000.0, 5.0, 150.0, "ms", "Release (middle band)"),
    n("mbc.makeup", "mbc", 0.0, 12.0, 0.5, 0.0, "dB", "Make-up gain"),
    b("comp.on", "comp", false, "Compressor"),
    n("comp.threshold", "comp", -50.0, 0.0, 0.5, -24.0, "dB", "Threshold"),
    n("comp.ratio", "comp", 1.0, 20.0, 0.1, 2.0, ":1", "Ratio"),
    n("comp.attack", "comp", 0.1, 100.0, 0.1, 20.0, "ms", "Attack"),
    n("comp.release", "comp", 10.0, 1000.0, 5.0, 200.0, "ms", "Release"),
    n("comp.makeup", "comp", 0.0, 18.0, 0.5, 0.0, "dB", "Make-up gain"),
    b("loud.on", "loud", false, "Loudness levelling"),
    n("loud.target", "loud", -30.0, -8.0, 0.5, -18.0, "LUFS", "Target"),
    n("loud.max", "loud", 0.0, 12.0, 0.5, 6.0, "dB", "Largest correction"),
    n("loud.speed", "loud", 0.5, 10.0, 0.5, 3.0, "s", "Reaction time"),
    n("out.gain", "out", -24.0, 12.0, 0.1, 0.0, "dB", "Output gain"),
    n("out.balance", "out", -100.0, 100.0, 1.0, 0.0, "%", "Balance (left − / right +)"),
    n("lim.ceiling", "out", -6.0, 0.0, 0.1, -1.0, "dB", "Limiter ceiling"),
    n("lim.release", "out", 10.0, 500.0, 5.0, 80.0, "ms", "Limiter release"),
];

pub const MAX_BANDS: usize = 32;

fn idx(key: &str) -> Option<usize> {
    SCHEMA.iter().position(|p| p.key == key)
}

/// The schema as text, one row per line, tab separated:
/// `g id title kind summary` and `p key group type min max step default unit title`.
pub fn schema_text() -> String {
    let mut s = format!("v\t{SCHEMA_VERSION}\n");
    for g in GROUPS {
        s += &format!("g\t{}\t{}\t{}\t{}\n", g.id, g.title, g.kind, g.summary);
    }
    for p in SCHEMA {
        let ty = match p.ty {
            Ty::Bool => "bool",
            Ty::Num => "num",
            Ty::Bands => "bands",
            Ty::Text => "text",
        };
        s += &format!(
            "p\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            p.key, p.group, ty, p.min, p.max, p.step, p.def, p.unit, p.title
        );
    }
    s
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Band {
    pub kind: Kind,
    pub f: f64,
    pub gain: f64,
    pub q: f64,
}

const KINDS: [(&str, Kind); 8] = [
    ("pk", Kind::Peak),
    ("ls", Kind::LowShelf),
    ("hs", Kind::HighShelf),
    ("lp", Kind::LowPass),
    ("hp", Kind::HighPass),
    ("no", Kind::Notch),
    ("bp", Kind::BandPass),
    ("ap", Kind::AllPass),
];

fn num(s: &str) -> Result<f64, String> {
    s.trim().parse::<f64>().ok().filter(|v| v.is_finite()).ok_or_else(|| format!("bad number '{s}'"))
}

fn bands(s: &str) -> Result<Vec<Band>, String> {
    let mut out = Vec::new();
    for item in s.split(',').filter(|i| !i.trim().is_empty()) {
        let f: Vec<&str> = item.split('/').collect();
        if f.len() != 4 {
            return Err(format!("bad band '{item}'"));
        }
        let kind = KINDS.iter().find(|(n, _)| *n == f[0].trim()).ok_or_else(|| format!("bad band type '{}'", f[0]))?.1;
        out.push(Band {
            kind,
            f: num(f[1])?.clamp(10.0, 96000.0),
            gain: num(f[2])?.clamp(-24.0, 24.0),
            q: num(f[3])?.clamp(0.1, 20.0),
        });
        if out.len() > MAX_BANDS {
            return Err("too many bands".into());
        }
    }
    Ok(out)
}

/// One profile: a value for every schema row.
#[derive(Clone, PartialEq, Debug)]
pub struct Settings {
    num: Vec<f64>,
    pub bands: Vec<Band>,
    pub ir: String,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { num: SCHEMA.iter().map(|p| p.def).collect(), bands: Vec::new(), ir: String::new() }
    }
}

impl Settings {
    /// Parses `key:value|key:value`. Missing keys keep their default, numbers are clamped to
    /// the schema range, unknown keys are an error.
    pub fn parse(spec: &str) -> Result<Settings, String> {
        let mut s = Settings::default();
        for tok in spec.split('|').filter(|t| !t.trim().is_empty()) {
            let (k, v) = tok.split_once(':').ok_or_else(|| format!("bad token '{tok}'"))?;
            let i = idx(k.trim()).ok_or_else(|| format!("unknown key '{k}'"))?;
            let p = &SCHEMA[i];
            match p.ty {
                Ty::Bool => s.num[i] = if num(v)? != 0.0 { 1.0 } else { 0.0 },
                Ty::Num => s.num[i] = num(v)?.clamp(p.min, p.max),
                Ty::Bands => s.bands = bands(v)?,
                Ty::Text => {
                    let v = v.trim();
                    if v.len() > 64 || !v.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_' || c == b'-' || c == b'.') {
                        return Err(format!("bad name '{v}'"));
                    }
                    s.ir = v.to_string();
                }
            }
        }
        Ok(s)
    }

    pub fn n(&self, key: &str) -> f64 {
        match idx(key) {
            Some(i) => self.num[i],
            None => {
                debug_assert!(false, "no schema row '{key}'");
                0.0
            }
        }
    }

    pub fn on(&self, key: &str) -> bool {
        self.n(key) != 0.0
    }

    /// The canonical text form; `parse` of it gives the same settings.
    pub fn spec(&self) -> String {
        let mut out = Vec::new();
        for (i, p) in SCHEMA.iter().enumerate() {
            out.push(match p.ty {
                Ty::Bool | Ty::Num => format!("{}:{}", p.key, self.num[i]),
                Ty::Bands => {
                    let b: Vec<String> = self
                        .bands
                        .iter()
                        .map(|b| format!("{}/{}/{}/{}", KINDS.iter().find(|(_, k)| *k == b.kind).map_or("pk", |k| k.0), b.f, b.gain, b.q))
                        .collect();
                    format!("{}:{}", p.key, b.join(","))
                }
                Ty::Text => format!("{}:{}", p.key, self.ir),
            });
        }
        out.join("|")
    }
}

#[derive(Clone, PartialEq, Debug)]
pub struct ConvP {
    pub ir: String,
    pub gen: u64,
    pub wet: f64,
    pub dry: f64,
}

/// Stage parameters for one sample rate. `None` or empty means the stage is off.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct Plan {
    pub trim: Option<f64>,
    pub conv: Option<ConvP>,
    pub eq: Vec<Coef>,
    pub dyneq: Vec<DynBand>,
    pub bass: Option<BassP>,
    pub colour: Option<ColourP>,
    pub spatial: Option<SpatialP>,
    pub mbc: Option<MbcP>,
    pub comp: Option<CompP>,
    pub loud: Option<LoudP>,
    pub lim: Option<LimP>,
    /// Gain applied ahead of the chain, in dB (0 or negative).
    pub pre_db: f64,
    /// Worst-case boost of the enabled stages, in dB.
    pub boost_db: f64,
    /// Frames the chain holds audio back.
    pub latency: u32,
    /// Enabled stages this mix point cannot run (they would delay a fast mixer).
    pub skipped: Vec<&'static str>,
}

impl Plan {
    pub fn active(&self) -> bool {
        self.lim.is_some()
    }
}

/// Largest gain of a cascade between 10 Hz and Nyquist, in dB.
pub fn peak_db(coefs: &[Coef], fs: f64) -> f64 {
    let mut peak = 0f64;
    let mut f = 10.0;
    while f < 0.499 * fs {
        let w = 2.0 * std::f64::consts::PI * f / fs;
        peak = peak.max(coefs.iter().map(|c| c.mag(w)).product::<f64>());
        f *= 1.012;
    }
    to_db(peak)
}

/// `ir` looks up an uploaded impulse response: its generation and peak gain at this rate.
/// `may_delay` is false on a fast mixer: the plan then holds no audio back.
pub fn plan(s: &Settings, fs: f64, may_delay: bool, ir: &mut dyn FnMut(&str) -> Option<(u64, f64)>) -> Plan {
    let mut p = Plan::default();
    if !s.on("on") {
        return p;
    }
    if s.on("eq.on") {
        p.eq = s.bands.iter().filter(|b| b.f < 0.49 * fs).map(|b| design(b.kind, fs, b.f, b.gain, b.q)).collect();
    }
    let mut cascade = p.eq.clone();
    let mut boost = 0f64;
    if s.on("dyneq.on") {
        for i in 1..=4 {
            let k = |x: &str| s.n(&format!("dyneq.{i}.{x}"));
            if k("range") != 0.0 && k("freq") < 0.49 * fs {
                p.dyneq.push(DynBand {
                    f: k("freq"),
                    q: k("q"),
                    threshold: k("threshold"),
                    range: k("range"),
                    attack_ms: k("attack"),
                    release_ms: k("release"),
                });
                if k("range") > 0.0 {
                    cascade.push(design(Kind::Peak, fs, k("freq"), k("range"), k("q")));
                }
            }
        }
    }
    if s.on("bass.on") && (s.n("bass.gain") > 0.0 || s.n("bass.harm") > 0.0 || s.n("bass.cut") > 0.0) {
        let shelf = if s.n("bass.gain") > 0.0 { design(Kind::LowShelf, fs, s.n("bass.freq"), s.n("bass.gain"), 0.7071) } else { Coef::UNITY };
        let harm = s.n("bass.harm") / 100.0;
        p.bass = Some(BassP { shelf, fc: s.n("bass.freq"), harm, cut: s.n("bass.cut") });
        cascade.push(shelf);
        // Peak gain of the overtones, from `stages::peak_probe`: they replace a removed
        // fundamental, and add at most 28 % at full amount on top of a kept one.
        if s.n("bass.cut") <= 0.0 {
            boost += to_db(1.0 + 0.35 * harm);
        }
    }
    boost += peak_db(&cascade, fs).max(0.0);
    let exc = if s.on("exc.on") { s.n("exc.amount") / 100.0 } else { 0.0 };
    let tube = s.on("tube.on").then(|| s.n("tube.drive") / 100.0);
    if exc > 0.0 || tube.is_some() {
        if may_delay {
            p.colour = Some(ColourP { exc, exc_fc: s.n("exc.freq"), tube });
            boost += to_db(1.0 + 0.5 * exc);
        } else {
            p.skipped.push("colour");
        }
    }
    let feed = if s.on("xfeed.on") { s.n("xfeed.level") } else { 0.0 };
    let w = |k: &str| if s.on("width.on") { s.n(k) / 100.0 } else { 1.0 };
    let (width, low, high) = (w("width.amount"), w("width.low"), w("width.high"));
    if feed > 0.0 || width != 1.0 || low != 1.0 || high != 1.0 {
        let (xlow, xhigh) = (s.n("width.xlow"), s.n("width.xhigh").min(0.45 * fs));
        p.spatial = Some(SpatialP { feed, fc: s.n("xfeed.freq"), width, low, high, xlow, xhigh });
        // A hard-panned source gains (1 + w) / 2.
        boost += to_db((1.0 + width.max(low).max(high)) / 2.0).max(0.0);
    }
    if s.on("conv.on") && !s.ir.is_empty() {
        if let Some((gen, gain)) = ir(&s.ir) {
            let wet = db(s.n("conv.wet"));
            let dry = s.n("conv.dry") / 100.0;
            p.conv = Some(ConvP { ir: s.ir.clone(), gen, wet, dry });
            boost += to_db(wet * gain + dry).max(0.0);
        }
    }
    if s.on("mbc.on") {
        p.mbc = Some(MbcP {
            xlow: s.n("mbc.xlow"),
            xhigh: s.n("mbc.xhigh").min(0.45 * fs),
            threshold: s.n("mbc.threshold"),
            ratio: s.n("mbc.ratio"),
            attack_ms: s.n("mbc.attack"),
            release_ms: s.n("mbc.release"),
            makeup: s.n("mbc.makeup"),
        });
    }
    if s.on("loud.on") {
        p.loud = Some(LoudP { target: s.n("loud.target"), max_db: s.n("loud.max"), tau_s: s.n("loud.speed") });
    }
    if s.on("comp.on") {
        p.comp = Some(CompP {
            threshold: s.n("comp.threshold"),
            ratio: s.n("comp.ratio"),
            attack_ms: s.n("comp.attack"),
            release_ms: s.n("comp.release"),
            makeup: s.n("comp.makeup"),
        });
    }
    let pre = if s.on("pre.auto") { -boost } else { s.n("pre.gain") };
    let out = s.n("out.gain");
    let any = !p.eq.is_empty()
        || !p.dyneq.is_empty()
        || p.bass.is_some()
        || p.colour.is_some()
        || p.spatial.is_some()
        || p.conv.is_some()
        || p.mbc.is_some()
        || p.comp.is_some()
        || p.loud.is_some();
    let bal = s.n("out.balance") / 100.0;
    if !any && pre == 0.0 && out == 0.0 && bal == 0.0 {
        return Plan { skipped: p.skipped, ..Plan::default() };
    }
    p.pre_db = pre;
    p.boost_db = boost;
    if pre != 0.0 {
        p.trim = Some(db(pre));
    }
    // A fixed delay: the look-ahead takes what the colour stage leaves of the budget.
    let lookahead = if may_delay {
        let budget = (DELAY_BUDGET_MS * 1e-3 * fs).round() as u32;
        let used = if p.colour.is_some() { colour::LATENCY as u32 } else { 0 };
        p.latency = budget.max(used + MIN_LOOKAHEAD);
        p.latency - used
    } else {
        0
    };
    let g = db(out);
    let gain = [g * (1.0 - bal.max(0.0)), g * (1.0 + bal.min(0.0))];
    p.lim = Some(LimP { gain, ceiling: db(s.n("lim.ceiling")), release_ms: s.n("lim.release"), lookahead });
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_is_consistent() {
        for (i, p) in SCHEMA.iter().enumerate() {
            assert!(GROUPS.iter().any(|g| g.id == p.group), "{}", p.key);
            assert!(SCHEMA.iter().skip(i + 1).all(|o| o.key != p.key), "duplicate {}", p.key);
            assert!(!p.key.contains(['|', ':', ';', '=', '\t']) && !p.title.contains('\t'));
            if p.ty == Ty::Num {
                assert!(p.min < p.max && p.step > 0.0 && p.def >= p.min && p.def <= p.max, "{}", p.key);
            }
        }
        assert_eq!(schema_text().lines().count(), 1 + GROUPS.len() + SCHEMA.len());
        // One switch per effect, named after it; the pages collapse an effect around it.
        for g in GROUPS {
            assert!(!g.summary.contains(['\t', '\n']));
            let switches: Vec<&str> = SCHEMA.iter().filter(|p| p.group == g.id && p.ty == Ty::Bool).map(|p| p.key).collect();
            match g.id {
                "main" => assert_eq!(switches, ["on", "pre.auto"]),
                "out" => assert!(switches.is_empty()),
                id => assert_eq!(switches, [format!("{id}.on")], "{id}"),
            }
        }
    }

    #[test]
    fn defaults_are_off_and_plan_reads_only_known_keys() {
        let mut none = |_: &str| None;
        assert_eq!(plan(&Settings::default(), 48000.0, true, &mut none), Plan::default());
        // Everything on: every key `plan` reads must exist (debug assertion in `n`).
        let all: Vec<String> = SCHEMA.iter().filter(|p| p.ty == Ty::Bool).map(|p| format!("{}:1", p.key)).collect();
        let s = Settings::parse(&format!("{}|eq.bands:pk/1000/6/1|conv.ir:room", all.join("|"))).unwrap();
        let mut ir = |_: &str| Some((1u64, 2.0));
        let p = plan(&s, 48000.0, true, &mut ir);
        assert!(p.active() && p.conv.is_some() && p.bass.is_some() && p.comp.is_some() && p.spatial.is_some());
        assert!(p.mbc.is_some() && p.loud.is_some() && p.dyneq.len() == 2);
        assert!(p.colour.is_some() && p.latency == 96);
    }

    #[test]
    fn fast_mixer_plans_hold_nothing_back() {
        let mut ir = |_: &str| Some((1u64, 2.0));
        let all: Vec<String> = SCHEMA.iter().filter(|p| p.ty == Ty::Bool).map(|p| format!("{}:1", p.key)).collect();
        let full = format!("{}|eq.bands:pk/1000/6/1|conv.ir:room|dyneq.1.range:4|dyneq.2.range:-6", all.join("|"));
        for spec in [full.as_str(), "on:1|tube.on:1", "on:1|eq.on:1|eq.bands:pk/100/3/1"] {
            let s = Settings::parse(spec).unwrap();
            for fs in [44100.0, 48000.0, 96000.0] {
                let fast = plan(&s, fs, false, &mut ir);
                assert_eq!(fast.latency, 0);
                assert!(fast.colour.is_none() && fast.lim.as_ref().is_none_or(|l| l.lookahead == 0));
                // The deep buffer's delay is the same whichever stages are on.
                let deep = plan(&s, fs, true, &mut ir);
                if !deep.active() {
                    continue;
                }
                assert_eq!(deep.latency, (DELAY_BUDGET_MS * 1e-3 * fs).round() as u32);
                let lim = deep.lim.as_ref().unwrap().lookahead;
                let colour = if deep.colour.is_some() { colour::LATENCY as u32 } else { 0 };
                assert_eq!(lim + colour, deep.latency);
            }
        }
        let tube = plan(&Settings::parse("on:1|tube.on:1").unwrap(), 48000.0, false, &mut ir);
        assert_eq!(tube.skipped, ["colour"]);
    }

    #[test]
    fn parse_round_trips_and_clamps() {
        let s = Settings::parse("on:1|eq.on:1|eq.bands:pk/1000/3.5/1.2,ls/80/40/0.7|bass.gain:99|conv.ir:a-b_1.wav").unwrap();
        assert_eq!(s.n("bass.gain"), 12.0);
        assert_eq!(s.bands[1].gain, 24.0);
        assert_eq!(Settings::parse(&s.spec()).unwrap(), s);
        assert!(Settings::parse("nope:1").is_err());
        assert!(Settings::parse("bass.gain:abc").is_err());
        assert!(Settings::parse("conv.ir:../x").is_err());
        assert!(Settings::parse("eq.bands:pk/1000/3").is_err());
    }

    #[test]
    fn balance_only_attenuates() {
        let mut none = |_: &str| None;
        let p = plan(&Settings::parse("on:1|out.balance:-25").unwrap(), 48000.0, false, &mut none);
        assert_eq!(p.lim.unwrap().gain, [1.0, 0.75]);
        let p = plan(&Settings::parse("on:1|out.balance:40").unwrap(), 48000.0, false, &mut none);
        assert_eq!(p.lim.unwrap().gain, [0.6, 1.0]);
    }

    #[test]
    fn automatic_headroom_cancels_the_boost() {
        let s = Settings::parse("on:1|eq.on:1|eq.bands:pk/1000/6/1.4").unwrap();
        let mut none = |_: &str| None;
        let p = plan(&s, 48000.0, false, &mut none);
        assert!((p.boost_db - 6.0).abs() < 0.05 && (p.pre_db + p.boost_db).abs() < 1e-9);
        assert!((p.trim.unwrap() - db(-p.boost_db)).abs() < 1e-12);
        // Cuts need no headroom and add no trim stage.
        let cut = Settings::parse("on:1|eq.on:1|eq.bands:pk/1000/-6/1.4").unwrap();
        let p = plan(&cut, 48000.0, false, &mut none);
        assert!(p.trim.is_none() && p.lim.is_some());
        // Master on with nothing enabled leaves the chain empty.
        assert_eq!(plan(&Settings::parse("on:1").unwrap(), 48000.0, true, &mut none), Plan::default());
    }
}
