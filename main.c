#include "X11/X.h"
#include <X11/Xlib.h>
#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/mman.h>

#include <x86intrin.h>

typedef uint64_t u64;
typedef uint32_t u32;
typedef uint16_t u16;
typedef uint8_t u8;

typedef int64_t i64;
typedef int32_t i32;
typedef int16_t i16;
typedef int8_t i8;

typedef float f32;
typedef double f64;

static clockid_t system_clock;
f32 nowf() {
    struct timespec ts;
    clock_gettime(system_clock,  &ts);
    f32 res = 0;
    res += ts.tv_nsec / 1000000000.0;
    res += ts.tv_sec;
    return res;
}

u32
MixColor(u8 r, u8 g, u8 b) {
    return b | (g << 8) | (r << 16);
}

typedef struct {
    f32 frame_delta;
    f32 last_time;
} FrameTimer;

FrameTimer
FrameTimer_Init(i32 fps_limit) {
    FrameTimer ft;
    ft.frame_delta = fps_limit == 0 ? 0.0 : 1.0 / fps_limit;
    ft.last_time = nowf();
    return ft;
}

f32
FrameTimer_NextFrame(FrameTimer* ft) {
    f32 cur_time = nowf();
    f32 delta;

    if (ft->last_time + ft->frame_delta > cur_time) {
        f32 f_delta = ft->last_time + ft->frame_delta - cur_time;
        struct timespec sleep_delta;
        sleep_delta.tv_sec = f_delta;
        sleep_delta.tv_nsec = (f_delta - (long)f_delta) * 1e9;
        delta = ft->frame_delta;
        nanosleep(&sleep_delta, 0);
    } else {
        delta = cur_time - ft->last_time;
    }

    ft->last_time = cur_time;
    return delta;
}

typedef struct {
    u8 keys[256];
} Keyboard;

typedef struct {
    u8 running;
    double speed_x;
    double speed_y;
    double pos_x;
    double pos_y;
    double width;
    double height;
} LogicState;

typedef struct {
    u8* data;
    i32 width;
    i32 height;
} RenderData;

RenderData
RenderData_Init(i32 width, i32 height) {
    void* mem = mmap(0, width * height * 4, PROT_READ | PROT_WRITE, MAP_POPULATE | MAP_PRIVATE | MAP_ANON, -1, 0);
    if (mem == (void*)-1) {
        printf("Error mmap: %d\n", errno);
        exit(1);
    }

    u8* data = mem;
    for (i64 i = 0; i < width * height * 4; i += 4) {
        data[i] = 0; // blue channel
        data[i + 1] = 0; // green channel
        data[i + 2] = 0; // red channel
        data[i + 3] = 0; // dead byte that is always zero
    }

    RenderData rd;
    rd.data = mem;
    rd.height = height;
    rd.width = width;
    return rd;
}

#define MAX_ITERS 100
u32 mandelbrot_pallete[MAX_ITERS]; // for scalar part
u32 mandelbrot_pallete2[MAX_ITERS]; // for vectorized part

#define ALIGN64 __attribute__((aligned(64)))

void
RenderData_MandelbrotIter(RenderData rd, f64 x0, f64 y0, f64 x1, f64 y1) {
    f64 v_width = x1 - x0;
    f64 v_height = y1 - y0;
    for (i32 row = 0; row < rd.height; row++) {
        for (i32 col = 0; col < rd.width;) {
            u32* dst = (u32*)rd.data + row * rd.width + col;

            if (col + 8 <= rd.width) {
                // fastpath with avx2
                f64 base_x = (f64)col / rd.width * v_width + x0;
                f64 diff_x = (1.0 / rd.width) * v_width;
                __m256 x_start = _mm256_set_ps(
                    base_x + diff_x * 0.0,
                    base_x + diff_x * 1.0,
                    base_x + diff_x * 2.0,
                    base_x + diff_x * 3.0,
                    base_x + diff_x * 4.0,
                    base_x + diff_x * 5.0,
                    base_x + diff_x * 6.0,
                    base_x + diff_x * 7.0);
                f64 base_y = (f64)row / rd.height * v_height + y0;
                f64 diff_y = (1.0 / rd.height) * v_height;
                __m256 y_start = _mm256_set_ps(
                    base_y + diff_y * 7.0, // 0 7
                    base_y + diff_y * 6.0, // 1 6
                    base_y + diff_y * 5.0, // 2 5
                    base_y + diff_y * 4.0, // 3 4
                    base_y + diff_y * 3.0, // 4 3
                    base_y + diff_y * 2.0, // 5 2
                    base_y + diff_y * 1.0, // 6 1
                    base_y + diff_y * 0.0); // 7 0

                __m256 x = x_start;
                __m256 y = y_start;

                __m256 stopper = _mm256_set1_ps(4.0);
                i32 i = 0;
                
                __m256 stored_iters = (__m256)_mm256_set1_epi32(0);

                while (i < MAX_ITERS) {
                    __m256 x2 = _mm256_mul_ps(x, x); // x2 = x * x
                    __m256 y2 = _mm256_mul_ps(y, y); // y2 = y * y
                    __m256 sum = _mm256_add_ps(x2, y2); // sum = x2 + y2

                    __m256 cmp = _mm256_cmp_ps(sum, stopper, _CMP_GT_OQ); // cmp = sum > 4
                    // cmp[i] == 0xFFFFFFFF if greater

                    __m256 iters = (__m256)_mm256_set1_epi32(i);
                    __m256 masked_iters = _mm256_and_ps(iters, cmp);
                    stored_iters = _mm256_or_ps(stored_iters, masked_iters);

                    __m256 tmp = _mm256_add_ps(_mm256_sub_ps(x2, y2), x_start);
                    __m256 xy = _mm256_mul_ps(x, y);
                    y = _mm256_add_ps(_mm256_add_ps(xy, xy), y_start);
                    x = tmp;

                    // reset x and y that are out of bounds
                    x = _mm256_andnot_ps(cmp, x);
                    y = _mm256_andnot_ps(cmp, y);
                    x_start = _mm256_andnot_ps(cmp, x_start);
                    y_start = _mm256_andnot_ps(cmp, y_start);

                    __m256 stop = _mm256_or_ps(x, y);
                    if (_mm256_testz_si256((__m256i)stop, (__m256i)stop)) { // early-return if everything is processed
                        break;
                    }

                    i++;
                }

                // __m256 cmp_x = _mm256_cmp_ps(x, _mm256_set1_ps(0), _CMP_NEQ_OQ);
                // __m256 cmp_y = _mm256_cmp_ps(y, _mm256_set1_ps(0), _CMP_NEQ_OQ);
                // __m256 cmp = _mm256_or_ps(cmp_x, cmp_y);
                // __m256 masked_idx = _mm256_and_ps(cmp, _mm256_set1_epi32(MAX_ITERS - 1));
                // stored_iters = _mm256_or_ps(stored_iters, masked_idx);

                f32 packed_iters[8] ALIGN64;
                _mm256_store_ps(packed_iters, stored_iters);
                
                for (i32 i = 0; i < 8; i++) { // may be optimizes with scatter ops, but they are avx512 only :(
                    i32 iters = ((i32*)packed_iters)[7 - i];
                    u32 color = mandelbrot_pallete2[iters];
                    // printf("idx: %d color %d\n", iters, color);
                    *(dst + i) = color;
                }

                col += 8;
                continue;
            }

            // calc mndlbrt value
            f32 x_start = (f32)col / rd.width * v_width + x0;
            f32 y_start = (f32)row / rd.height * v_height + y0;
            f32 x = x_start;
            f32 y = y_start;
            u32 iters = 0;
            while (1) {
                f32 x2 = x * x;
                f32 y2 = y * y;
                if (x2 + y2 >= 4 || iters + 1 >= MAX_ITERS) {
                    break;
                }
                f32 tmp = x2 - y2 + x_start;
                y = 2 * x * y + y_start;
                x = tmp;
                iters++;
            }

            u32 color = mandelbrot_pallete[iters];
            *dst = color;

            col++;
        }
    }
}

void
RenderData_Fill(RenderData rd, u32 pix) {
    for (i32 row = 0; row < rd.height; row++) {
        for (i32 col = 0; col < rd.width; col++) {
            u32* dst = (u32*)rd.data + row * rd.width + col;
            *dst = pix;
        }
    }
}

void
RenderData_Deinit(RenderData rd) {
    munmap(rd.data, rd.height * rd.width * 4);
}

i32 main() {
    clock_getcpuclockid(getpid(), &system_clock);

    for (i32 i = 0; i < MAX_ITERS; i++) {
        mandelbrot_pallete[i] = MixColor(0, (f32)i / MAX_ITERS * 255, 0);
    }
    for (i32 i = 0; i < MAX_ITERS; i++) {
        mandelbrot_pallete2[i] = MixColor(0, 0, (f32)i / MAX_ITERS * 255);
    }

    Display* d = XOpenDisplay(0);
    if (d == 0) {
        printf("Display opening failed\n");
        return 1;
    }
    i32 s = DefaultScreen(d);
    Window root = RootWindow(d, s);
    Window w = XCreateSimpleWindow(d, root, 10, 10, 800, 600, 1,
                           BlackPixel(d, s), WhitePixel(d, s));
    GC gc = DefaultGC(d, s);
    Visual* vis = DefaultVisual(d, 0);

    i64 mask = KeyPressMask | KeyReleaseMask | StructureNotifyMask;
    XSelectInput(d, w, mask);
    XMapWindow(d, w);
    XFlush(d);

    u64 frame_counter = 0;
    i32 fps_limit = 120;
    FrameTimer ft = FrameTimer_Init(fps_limit);

    RenderData rd = RenderData_Init(1200, 800);
    XImage *image  = XCreateImage(d, vis, 24, ZPixmap, 0, (char*) rd.data, rd.width, rd.height, 32, 0);
    Pixmap pm = XCreatePixmap(d, w, rd.width, rd.height, 24);
    RenderData_Fill(rd, MixColor(100, 0, 0));

    Keyboard keyboard = {0};
    LogicState state = {0};
    state.pos_x = -2;
    state.pos_y = 1;
    state.width = 3;
    state.height = 2;
    state.speed_x = 1.0;
    state.speed_y = 1.0;

    state.running = 1;
    while (state.running) {
        XEvent e;
        while (state.running && XCheckWindowEvent(d, w, mask, &e)) {
            if (e.type == KeyPress) {
                keyboard.keys[e.xkey.keycode] = 1;
            }
            if (e.type == KeyRelease) {
                keyboard.keys[e.xkey.keycode] = 0;
            }
            if (e.type == ConfigureNotify) {
                // TODO(optimize): just populate extra pages for buffer
                XFreePixmap(d, pm);
                RenderData_Deinit(rd);
                rd = RenderData_Init(e.xconfigure.width, e.xconfigure.height);
                image = XCreateImage(d, vis, 24, ZPixmap, 0, (char*) rd.data, rd.width, rd.height, 32, 0);
                pm = XCreatePixmap(d, w, rd.width, rd.height, 24);
            }
        }

        f32 dt = FrameTimer_NextFrame(&ft);
        frame_counter++;

        if (keyboard.keys[25]) { // W
            state.pos_y += state.speed_y * dt;
        }
        if (keyboard.keys[38]) { // A
            state.pos_x -= state.speed_x * dt;
        }
        if (keyboard.keys[39]) { // S
            state.pos_y -= state.speed_y * dt;
        }
        if (keyboard.keys[40]) { // D
            state.pos_x += state.speed_x * dt;
        }
        if (keyboard.keys[27]) { // R
            state.pos_x += state.width * 0.1 / 2.0;
            state.pos_y -= state.height * 0.1 / 2.0;
            state.width *= 0.9;
            state.height *= 0.9;
            state.speed_x *= 0.9;
            state.speed_y *= 0.9;
        }
        if (keyboard.keys[41]) { // F
            state.pos_x -= state.width * (0.1 / 0.9) / 2.0;
            state.pos_y += state.height * (0.1 / 0.9) / 2.0;
            state.width /= 0.9;
            state.height /= 0.9;
            state.speed_x /= 0.9;
            state.speed_y /= 0.9;
        }
        if (keyboard.keys[24] || keyboard.keys[9]) { // Q || Esc
            state.running = 0;
            break;
        }

        // RenderData_Fill(rd, MixColor(frame_counter, 0, 0));
        RenderData_MandelbrotIter(rd, state.pos_x, state.pos_y, state.pos_x + state.width, state.pos_y - state.height);
        // XPutImage(d, pm, gc, image, 0,0,0,0, rd.width, rd.height);
        XPutImage(d, w, gc, image, 0,0,0,0, rd.width, rd.height);
        XSync(d, 0);
    }
}
