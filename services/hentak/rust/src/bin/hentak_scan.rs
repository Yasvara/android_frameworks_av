//! hentak_scan <package> [user]: builds the package's sound catalog from its own files
//! (/data/media/<user>/Android/{data,obb}/<package>) into /data/misc/hentak/<package>.htc.
//! Exits at once when the files are unchanged since the last catalog. Started by init when
//! system_server sets sys.yasvara.hentak.scan=<package>[/<nonce>].

use hentak::catalog::Catalog;
use std::path::PathBuf;

const OUT: &str = "/data/misc/hentak";

fn log(msg: &str) {
    #[cfg(target_os = "android")]
    {
        extern "C" {
            fn __android_log_write(prio: i32, tag: *const std::ffi::c_char, text: *const std::ffi::c_char) -> i32;
        }
        let text = std::ffi::CString::new(msg).unwrap_or_default();
        // SAFETY: both strings are NUL-terminated and outlive the call. 4 = ANDROID_LOG_INFO.
        unsafe { __android_log_write(4, c"hentak_scan".as_ptr(), text.as_ptr()) };
    }
    eprintln!("{msg}");
}

/// sys.yasvara.hentak.scan, through bionic.
fn property() -> Option<String> {
    #[cfg(target_os = "android")]
    {
        extern "C" {
            fn __system_property_get(name: *const std::ffi::c_char, value: *mut std::ffi::c_char) -> i32;
        }
        let mut v = [0 as std::ffi::c_char; 92];
        // SAFETY: PROP_VALUE_MAX is 92 bytes, NUL included.
        let n = unsafe { __system_property_get(c"sys.yasvara.hentak.scan".as_ptr(), v.as_mut_ptr()) };
        if n > 0 {
            // SAFETY: bionic NUL-terminates the value.
            return Some(unsafe { std::ffi::CStr::from_ptr(v.as_ptr()) }.to_string_lossy().into_owned());
        }
    }
    None
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let arg = args.get(1).cloned().or_else(property);
    let Some(pkg) = arg.map(|a| a.split('/').next().unwrap_or("").to_string()).filter(|p| {
        !p.is_empty() && p.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'.' || c == b'_')
    }) else {
        log("no package");
        std::process::exit(2);
    };
    let user: u32 = args.get(2).and_then(|u| u.parse().ok()).unwrap_or(0);
    let roots: Vec<PathBuf> = ["data", "obb"].iter().map(|d| PathBuf::from(format!("/data/media/{user}/Android/{d}/{pkg}"))).collect();
    let out = PathBuf::from(format!("{OUT}/{pkg}.htc"));
    let previous = std::fs::read(&out).ok().and_then(|b| Catalog::from_bytes(&b)).map(|c| c.fingerprint);
    let started = std::time::Instant::now();
    let threads = std::thread::available_parallelism().map_or(4, |n| n.get().min(4));
    match hentak::scan::scan(&roots, previous, threads, 1500) {
        None => log(&format!("{pkg}: unchanged")),
        Some(s) => {
            let secs = started.elapsed().as_secs_f32();
            match s.catalog {
                Some(c) => {
                    let tmp = out.with_extension("tmp");
                    use std::os::unix::fs::PermissionsExt;
                    // 0640, group audio (the directory's setgid): audioserver reads it.
                    let written = std::fs::write(&tmp, c.to_bytes())
                        .and_then(|_| std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o640)))
                        .and_then(|_| std::fs::rename(&tmp, &out));
                    if written.is_err() {
                        log(&format!("{pkg}: cannot write {}", out.display()));
                        std::process::exit(1);
                    }
                    log(&format!(
                        "{pkg}: {} containers, {} sounds, {} decoded in {secs:.1} s; onsets per label {:?}; {} music tracks, {} prints",
                        s.containers,
                        s.sounds,
                        s.decoded,
                        c.sounds,
                        s.music_tracks,
                        c.music.as_ref().map_or(0, |m| m.prints())
                    ));
                }
                None => {
                    // Nothing readable (encrypted paks, no labels): no catalog, v2 stays.
                    let _ = std::fs::remove_file(&out);
                    log(&format!("{pkg}: {} containers, {} sounds: no catalog ({secs:.1} s)", s.containers, s.sounds));
                }
            }
        }
    }
}
