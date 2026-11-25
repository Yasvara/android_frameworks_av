//! A game's own sound calls as evidence for its hits. framefx/sounds, inside the game, records the
//! events it posts into a ring this side maps read-only (system_server hands it over). An onset
//! right after the game started a gun, a blast or a skill plays; one after only music, UI or a
//! voice line, or after no new event at all while the game is posting and nothing that vibrates
//! has started lately (a beat of the music, an ambience loop), stays quiet. An event this side
//! can't name leaves the hit to the tagger, and so does everything while the game posts nothing.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{fence, AtomicU64, Ordering};

// Mirrors frameworks/libs/framefx/sounds/rust/src/ring.rs.
const MAGIC: u32 = u32::from_le_bytes(*b"FXSN");
const VERSION: u32 = 1;
const HEADER: usize = 4096;
const RECORD: usize = 128;
const TEXT: usize = 88;
const POST: u32 = 1;
const NAME: u32 = 8;

/// Events this long before an onset can be its sound: the engine's frame, its buffer and the
/// track's.
const BEFORE_NS: u64 = 400_000_000;
/// ...and this long after it: the two clocks' jitter.
const AFTER_NS: u64 = 30_000_000;
/// After a sound that vibrates starts, onsets with no event of their own may still be its own (a
/// long skill), so the tagger decides them for this long.
const HOLD_NS: u64 = 2_500_000_000;
/// A game silent this long posts nothing this side can use.
const LIVE_NS: u64 = 30_000_000_000;
const KEEP_NS: u64 = HOLD_NS + BEFORE_NS;

/// What a sound is, in the tagger's label order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Class {
    Gun,
    Explosion,
    Impact,
    Swing,
    Skill,
    Fire,
    Footstep,
    Engine,
    Ui,
    Voice,
    Music,
    Ambience,
}

impl Class {
    pub fn vibrates(self) -> bool {
        matches!(self, Class::Gun | Class::Explosion | Class::Impact | Class::Swing | Class::Skill | Class::Fire)
    }

    /// Which of several starting at once makes the hit.
    fn rank(self) -> u8 {
        match self {
            Class::Explosion => 6,
            Class::Gun => 5,
            Class::Impact => 4,
            Class::Fire => 3,
            Class::Skill => 2,
            Class::Swing => 1,
            _ => 0,
        }
    }
}

/// What an event name says, by its words, first what the sound is for (a voice line, the music,
/// the UI), then what it depicts, so a voiced attack stays a voice line. None for a name that
/// says neither; `Some(None)` for a call that starts no sound (stop, set, mute).
pub fn classify(name: &str) -> Option<Option<Class>> {
    let lower = name.to_ascii_lowercase();
    let words: Vec<&str> = lower
        .split(|c: char| !c.is_ascii_alphanumeric())
        .map(|w| w.trim_end_matches(|c: char| c.is_ascii_digit()))
        .filter(|w| !w.is_empty())
        .collect();
    let any = |list: &[&str]| words.iter().any(|w| list.contains(w));
    let prefix = |list: &[&str]| words.iter().any(|w| list.iter().any(|p| w.starts_with(p)));
    if any(&["stop", "pause", "resume", "mute", "unmute", "set", "reset", "init", "load", "unload", "rtpc", "state"]) {
        return Some(None);
    }
    let class = if any(&["vo", "voice", "vox", "speech", "talk", "chat"]) || prefix(&["dialog"]) {
        Class::Voice
    } else if any(&["bgm", "music", "mus", "jukebox", "song", "stinger"]) || prefix(&["music"]) {
        Class::Music
    } else if any(&["ui", "menu", "click", "button", "btn", "hud", "popup", "page", "tab", "gacha", "reward", "notify", "notice", "tips", "login", "toast"]) {
        Class::Ui
    } else if any(&["amb", "env", "weather", "rain", "birds", "insects"]) || prefix(&["ambien", "environment"]) {
        Class::Ambience
    } else if any(&["foot", "step", "steps", "walk", "run", "jog", "sprint", "land", "jump", "foley", "cloth", "move", "movement", "locomotion"]) || prefix(&["footstep"]) {
        Class::Footstep
    } else if any(&["engine", "vehicle", "car", "motor"]) {
        Class::Engine
    } else if any(&["blast", "bomb", "grenade", "boom", "detonate"]) || prefix(&["explo"]) {
        Class::Explosion
    } else if any(&["flame", "flamethrower", "burn", "torch"]) {
        Class::Fire
    } else if any(&["gun", "shot", "shoot", "rifle", "pistol", "smg", "sniper", "shotgun", "bullet", "fire", "muzzle"]) || prefix(&["gunshot"]) {
        Class::Gun
    } else if any(&["hit", "impact", "dmg", "damage", "punch", "smash", "slam", "crash", "crit", "break", "shatter", "collide"]) {
        Class::Impact
    } else if any(&["swing", "swish", "whoosh", "woosh", "slash", "swipe"]) {
        Class::Swing
    } else if any(&["skill", "ult", "ultimate", "burst", "attack", "atk", "cast", "spell", "combat", "battle", "elemental", "charge", "ability"]) {
        Class::Skill
    } else {
        return None;
    };
    Some(Some(class))
}

/// What the game's calls say about one onset.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Evidence {
    /// It started this just before.
    Started(Class),
    /// It started only sounds that don't vibrate, or nothing while nothing that does plays.
    Quiet,
    /// The tagger decides.
    Unknown,
}

/// The class of a post, or that it is unnamed here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Heard {
    Class(Class),
    Unnamed,
}

pub struct Sounds {
    base: *const u8,
    len: usize,
    unmap: bool,
    capacity: u64,
    next: u64,
    /// id → class, from the names the game hashed and the posts that named their event.
    names: HashMap<u32, Option<Class>>,
    posts: VecDeque<(u64, Heard)>,
    last_post: u64,
    last_vibrating: u64,
    /// Posts read, and of those named here.
    pub read: u64,
    pub named: u64,
}

// SAFETY: the mapping is read-only here and owned by this value; one thread polls it at a time.
unsafe impl Send for Sounds {}

extern "C" {
    fn munmap(addr: *mut u8, len: usize) -> i32;
}

impl Sounds {
    /// Adopts a read-only mapping of a game's ring; None if it isn't one (the caller keeps it then).
    ///
    /// # Safety
    /// `base` maps `len` readable bytes, page aligned, that nobody else unmaps. With `unmap`, the
    /// value munmaps it when dropped.
    pub unsafe fn adopt(base: *const u8, len: usize, unmap: bool) -> Option<Self> {
        if base.is_null() || len < HEADER {
            return None;
        }
        // SAFETY: the header page is inside the mapping.
        let word = |at: usize| unsafe { std::ptr::read_volatile(base.add(at) as *const u32) };
        let (capacity, record) = (word(8) as usize, word(12) as usize);
        if word(0) != MAGIC || word(4) != VERSION || record != RECORD || !capacity.is_power_of_two() || HEADER + capacity * RECORD > len {
            return None;
        }
        Some(Self {
            base,
            len,
            unmap,
            capacity: capacity as u64,
            next: 0,
            names: HashMap::new(),
            posts: VecDeque::new(),
            last_post: 0,
            last_vibrating: 0,
            read: 0,
            named: 0,
        })
    }

    fn head(&self) -> u64 {
        // SAFETY: offset 64 of the header, 8-aligned in a page-aligned mapping.
        unsafe { &*(self.base.add(64) as *const AtomicU64) }.load(Ordering::Acquire)
    }

    /// Record `i`, unless a writer is in it or has moved past it.
    fn record(&self, i: u64) -> Option<(u64, u32, u32, [u8; TEXT])> {
        let at = HEADER + (i & (self.capacity - 1)) as usize * RECORD;
        // SAFETY: the slot is inside the mapping; a torn read is caught by the sequence check.
        unsafe {
            let p = self.base.add(at);
            let seq = &*(p as *const AtomicU64);
            if seq.load(Ordering::Acquire) != i + 1 {
                return None;
            }
            let t = std::ptr::read_volatile(p.add(8) as *const u64);
            let kind = std::ptr::read_volatile(p.add(16) as *const u32);
            let a = std::ptr::read_volatile(p.add(20) as *const u32);
            let mut text = [0u8; TEXT];
            for (k, b) in text.iter_mut().enumerate() {
                *b = std::ptr::read_volatile(p.add(40 + k));
            }
            fence(Ordering::Acquire);
            (seq.load(Ordering::Relaxed) == i + 1).then_some((t, kind, a, text))
        }
    }

    /// Posts at the given times (CLOCK_MONOTONIC ns), with no ring behind them.
    #[cfg(test)]
    pub fn from_posts(posts: &[(u64, &str)]) -> Self {
        let mut s = Self {
            base: std::ptr::null(),
            len: 0,
            unmap: false,
            capacity: 1,
            next: 0,
            names: HashMap::new(),
            posts: VecDeque::new(),
            last_post: 0,
            last_vibrating: 0,
            read: 0,
            named: 0,
        };
        for &(t, name) in posts {
            s.post(t, 0, name);
        }
        s
    }

    /// Reads what the game posted since the last call.
    pub fn poll(&mut self) {
        if self.base.is_null() {
            return;
        }
        let head = self.head();
        let mut i = self.next.max(head.saturating_sub(self.capacity));
        while i < head {
            if let Some((t, kind, a, text)) = self.record(i) {
                let n = text.iter().position(|&c| c == 0).unwrap_or(TEXT);
                let name = std::str::from_utf8(&text[..n]).unwrap_or("");
                match kind {
                    NAME if !name.is_empty() => {
                        if let Some(c) = classify(name) {
                            self.names.insert(a, c);
                        }
                    }
                    POST => self.post(t, a, name),
                    _ => {}
                }
            }
            i += 1;
        }
        self.next = head;
        let newest = self.last_post;
        while self.posts.front().is_some_and(|p| p.0 + KEEP_NS < newest) {
            self.posts.pop_front();
        }
    }

    fn post(&mut self, t: u64, id: u32, name: &str) {
        let class = if name.is_empty() {
            self.names.get(&id).copied()
        } else {
            let c = classify(name);
            if let Some(c) = c {
                self.names.insert(id, c);
            }
            c
        };
        self.read += 1;
        let heard = match class {
            Some(None) => return, // a stop or a setting: no sound starts
            Some(Some(c)) => {
                self.named += 1;
                Heard::Class(c)
            }
            None => Heard::Unnamed,
        };
        self.last_post = self.last_post.max(t);
        if matches!(heard, Heard::Class(c) if c.vibrates()) {
            self.last_vibrating = self.last_vibrating.max(t);
        }
        self.posts.push_back((t, heard));
    }

    /// Whether the game posted lately, as of `now` (CLOCK_MONOTONIC ns).
    pub fn live(&self, now: u64) -> bool {
        self.last_post != 0 && now < self.last_post + LIVE_NS
    }

    /// The evidence for an onset at `onset` (CLOCK_MONOTONIC ns).
    pub fn evidence(&self, onset: u64) -> Evidence {
        if !self.live(onset) {
            return Evidence::Unknown;
        }
        let (from, to) = (onset.saturating_sub(BEFORE_NS), onset + AFTER_NS);
        let mut best: Option<Class> = None;
        let (mut unnamed, mut quiet) = (false, false);
        for &(_, heard) in self.posts.iter().filter(|p| p.0 >= from && p.0 <= to) {
            match heard {
                Heard::Class(c) if c.vibrates() => {
                    if best.is_none_or(|b| c.rank() > b.rank()) {
                        best = Some(c);
                    }
                }
                Heard::Class(_) => quiet = true,
                Heard::Unnamed => unnamed = true,
            }
        }
        match best {
            Some(c) => Evidence::Started(c),
            None if unnamed => Evidence::Unknown,
            None if quiet => Evidence::Quiet,
            None if self.last_vibrating != 0 && onset < self.last_vibrating + HOLD_NS => Evidence::Unknown,
            None => Evidence::Quiet,
        }
    }
}

impl Drop for Sounds {
    fn drop(&mut self) {
        if self.unmap {
            // SAFETY: the mapping this value adopted with `unmap`, unused from here on.
            unsafe { munmap(self.base as *mut u8, self.len) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CAP: usize = 64;

    /// A ring as framefx/sounds lays it out, in 8-aligned memory.
    struct Fake {
        words: Vec<u64>,
        head: u64,
    }

    impl Fake {
        fn new() -> Self {
            let mut f = Fake { words: vec![0; (HEADER + CAP * RECORD) / 8], head: 0 };
            let b = f.bytes();
            b[0..4].copy_from_slice(&MAGIC.to_le_bytes());
            b[4..8].copy_from_slice(&VERSION.to_le_bytes());
            b[8..12].copy_from_slice(&(CAP as u32).to_le_bytes());
            b[12..16].copy_from_slice(&(RECORD as u32).to_le_bytes());
            f
        }

        fn bytes(&mut self) -> &mut [u8] {
            // SAFETY: plain integers viewed as their bytes.
            unsafe { std::slice::from_raw_parts_mut(self.words.as_mut_ptr() as *mut u8, self.words.len() * 8) }
        }

        fn push(&mut self, t_ms: u64, kind: u32, a: u32, text: &str) {
            let i = self.head;
            let at = HEADER + (i as usize % CAP) * RECORD;
            let b = self.bytes();
            b[at + 8..at + 16].copy_from_slice(&(t_ms * 1_000_000).to_le_bytes());
            b[at + 16..at + 20].copy_from_slice(&kind.to_le_bytes());
            b[at + 20..at + 24].copy_from_slice(&a.to_le_bytes());
            b[at + 40..at + 40 + TEXT].fill(0);
            b[at + 40..at + 40 + text.len()].copy_from_slice(text.as_bytes());
            b[at..at + 8].copy_from_slice(&(i + 1).to_le_bytes());
            self.head += 1;
            let h = self.head;
            self.bytes()[64..72].copy_from_slice(&h.to_le_bytes());
        }

        fn sounds(&mut self) -> Sounds {
            let len = self.words.len() * 8;
            // SAFETY: the fake outlives the reader in each test; nothing unmaps it.
            unsafe { Sounds::adopt(self.words.as_ptr() as *const u8, len, false) }.unwrap()
        }
    }

    const MS: u64 = 1_000_000;

    #[test]
    fn names_say_what_a_sound_is_for_first() {
        assert_eq!(classify("Ev_sfx_ui_global_common_click"), Some(Some(Class::Ui)));
        assert_eq!(classify("Ev_bgm_menu_login"), Some(Some(Class::Music)));
        assert_eq!(classify("Ev_vo_battle_attack_01"), Some(Some(Class::Voice)));
        assert_eq!(classify("Play_Weapon_AK47_Fire_1P"), Some(Some(Class::Gun)));
        assert_eq!(classify("Play_Grenade_Explosion"), Some(Some(Class::Explosion)));
        assert_eq!(classify("Ev_sfx_battle_hit_heavy"), Some(Some(Class::Impact)));
        assert_eq!(classify("Ev_sfx_avatar_skill_ult02"), Some(Some(Class::Skill)));
        assert_eq!(classify("Ev_sfx_menu_login_loop_01_stop"), Some(None));
        assert_eq!(classify("Ev_global_set_bgm_state"), Some(None));
        assert_eq!(classify("Ev_sfx_cinematic_cutscene"), None);
    }

    #[test]
    fn a_skill_plays_a_beat_stays_quiet() {
        let mut f = Fake::new();
        f.push(1000, POST, 1, "Ev_bgm_battle");
        f.push(5000, POST, 2, "Ev_sfx_avatar_skill_e");
        let mut s = f.sounds();
        s.poll();
        assert_eq!(s.evidence(5080 * MS), Evidence::Started(Class::Skill));
        // A beat with no event of its own, long after the skill.
        assert_eq!(s.evidence(9000 * MS), Evidence::Quiet);
        // Within the skill's hold: the tagger decides.
        assert_eq!(s.evidence(6500 * MS), Evidence::Unknown);
    }

    #[test]
    fn ids_named_by_the_game_and_unnamed_ones() {
        let mut f = Fake::new();
        f.push(10, NAME, 77, "Ev_sfx_ui_global_common_click");
        f.push(2000, POST, 77, "");
        f.push(4000, POST, 88, "");
        let mut s = f.sounds();
        s.poll();
        assert_eq!(s.evidence(2050 * MS), Evidence::Quiet);
        assert_eq!(s.evidence(4050 * MS), Evidence::Unknown);
        assert_eq!((s.read, s.named), (2, 1));
    }

    #[test]
    fn a_silent_game_leaves_it_to_the_tagger() {
        let mut f = Fake::new();
        f.push(1000, POST, 1, "Ev_bgm_menu");
        let mut s = f.sounds();
        s.poll();
        assert_eq!(s.evidence(60_000 * MS), Evidence::Unknown);
    }

    #[test]
    fn follows_the_ring_across_polls_and_wraps() {
        let mut f = Fake::new();
        let mut s = f.sounds();
        for k in 0..(CAP as u64 + 10) {
            f.push(1000 + k, POST, 5, "Ev_sfx_ui_tab");
        }
        s.poll();
        assert_eq!(s.read, CAP as u64);
        f.push(3000, POST, 6, "Play_Gun_Fire");
        s.poll();
        assert_eq!(s.evidence(3050 * MS), Evidence::Started(Class::Gun));
    }
}
