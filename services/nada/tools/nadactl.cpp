/*
 * SPDX-FileCopyrightText: The Yasvara Project
 * SPDX-License-Identifier: Apache-2.0
 */

// Test and diagnostics front for the Nada keys of audioserver, for a root shell:
//   nadactl set "nada_cfg=spk|on:1|eq.on:1|eq.bands:pk/1000/-6/1.4"
//   nadactl get nada_dump | nada_schema | nada_active

#include <cstdio>
#include <cstring>
#include <unistd.h>

#include <binder/ProcessState.h>
#include <media/AudioSystem.h>
#include <private/android_filesystem_config.h>

using namespace android;

int main(int argc, char** argv) {
    if (argc != 3 || (strcmp(argv[1], "set") != 0 && strcmp(argv[1], "get") != 0)) {
        fprintf(stderr, "usage: nadactl set <key=value[;key=value]> | get <key>\n");
        return 2;
    }
    // audioserver takes these keys from system_server's uid only.
    if (getuid() == 0 && (setgid(AID_SYSTEM) != 0 || setuid(AID_SYSTEM) != 0)) return 1;
    ProcessState::self()->startThreadPool();
    if (strcmp(argv[1], "set") == 0) {
        const status_t s = AudioSystem::setParameters(AUDIO_IO_HANDLE_NONE, String8(argv[2]));
        if (s != NO_ERROR) fprintf(stderr, "setParameters: %d\n", s);
        return s == NO_ERROR ? 0 : 1;
    }
    fputs(AudioSystem::getParameters(AUDIO_IO_HANDLE_NONE, String8(argv[2])).c_str(), stdout);
    return 0;
}
