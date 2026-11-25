//! Hentak v3: game haptics from game audio.
//!
//! Split by thread. The mixer thread only downmixes into a [`ring::Ring`]. A worker thread runs,
//! per track, the v2 transient detector ([`tap2`]) and, for a game with a catalog, the onset
//! classifier: log-mel frames ([`mel`]) of the [`front::Front`]-decimated signal, onsets
//! ([`onset`]), their embedding ([`model`]) and the game's [`catalog::Catalog`]. Taught sounds
//! ([`verify`], [`engine`]) are the next layer.
//!
//! Catalogs (feature `scan`, the `hentak_scan` tool): [`bank`] reads Wwise soundbanks and
//! packages, [`wwise`] turns their Wwise Vorbis into standard Vorbis packets, [`vorbis`] decodes
//! them and [`scan`] trains the catalog on the onsets in every sound.

#![deny(unsafe_op_in_unsafe_fn)]

#[cfg(feature = "scan")]
pub mod bank;
pub mod bits;
pub mod catalog;
pub mod engine;
pub mod ffi;
pub mod front;
pub mod mel;
pub mod model;
pub mod music;
pub mod onset;
pub mod ring;
pub mod rt;
#[cfg(feature = "scan")]
pub mod scan;
pub mod tagger;
pub mod tap2;
pub mod verify;
#[cfg(feature = "scan")]
pub mod vorbis;
#[cfg(feature = "scan")]
pub mod wwise;
