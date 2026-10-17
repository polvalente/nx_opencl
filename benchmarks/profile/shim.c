// Profiling shim for libOpenCL. Not part of the library.
//
// Point OPENCL_DYLIB_PATH at the built shim. It forwards every OpenCL call to the
// real library (NXCL_REAL_OPENCL) and, for the calls below, keeps counters in a
// memory-mapped file (NXCL_PROF_FILE) that profile.exs reads between calls:
//
//   kernel   device time of each kernel launch, from CL_QUEUE_PROFILING_ENABLE events
//   upload   wall time of clCreateBuffer with CL_MEM_COPY_HOST_PTR
//   read     wall time of clEnqueueReadBuffer
//   alloc    wall time of clCreateBuffer without host data
//   build    wall time of clBuildProgram
//   enqueue  wall time of clEnqueueNDRangeKernel
//   finish   wall time of clFinish (waiting for the device)
//
// Layout of the file: u64 header[16] (upload_ns, upload_n, read_ns, read_n, alloc_ns,
// alloc_n, build_ns, build_n, enqueue_ns, enqueue_n, finish_ns, finish_n, 0 x4), then MAX_SLOTS slots of {char name[56]; u64 count; u64 ns}.

#include <dlfcn.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

typedef void *H;
typedef int32_t cl_int;
typedef uint32_t cl_uint;
typedef uint64_t cl_ulong;
typedef uint64_t cl_mem_flags;
typedef uint64_t cl_bitfield;
#define CL_PROFILING_COMMAND_START 0x1282
#define CL_PROFILING_COMMAND_END 0x1283
#define CL_MEM_COPY_HOST_PTR (1ull << 5)
#define CL_KERNEL_FUNCTION_NAME 0x1190
#define MAX_SLOTS 512
#define MAX_PENDING 65536
#define MAX_KERNELS 65536

typedef struct { char name[56]; uint64_t count, ns; } Slot;
typedef struct { uint64_t h[16]; Slot slots[MAX_SLOTS]; } Stats;

static Stats *stats;
static void *real;
static struct { H kernel; int slot; } kmap[MAX_KERNELS];
static int nk;
static struct { H event; int slot; } pending[MAX_PENDING];
static int np;

static uint64_t now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (uint64_t)t.tv_sec * 1000000000ull + t.tv_nsec;
}

static void init(void) {
  if (stats) return;
  const char *path = getenv("NXCL_REAL_OPENCL");
  real = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  const char *file = getenv("NXCL_PROF_FILE");
  int fd = open(file, O_RDWR | O_CREAT, 0644);
  ftruncate(fd, sizeof(Stats));
  stats = mmap(0, sizeof(Stats), PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
}

#define REAL(name) ((typeof(&name))dlsym(real, #name))

static int slot_for(const char *name) {
  for (int i = 0; i < MAX_SLOTS; i++) {
    if (stats->slots[i].name[0] == 0) {
      strncpy(stats->slots[i].name, name, 55);
      return i;
    }
    if (strncmp(stats->slots[i].name, name, 55) == 0) return i;
  }
  return MAX_SLOTS - 1;
}

H clCreateKernel(H program, const char *name, cl_int *err);
H clCreateKernel(H program, const char *name, cl_int *err) {
  init();
  H k = ((H(*)(H, const char *, cl_int *))dlsym(real, "clCreateKernel"))(program, name, err);
  if (k && nk < MAX_KERNELS) {
    kmap[nk].kernel = k;
    kmap[nk].slot = slot_for(name);
    nk++;
  }
  return k;
}

cl_int clEnqueueNDRangeKernel(H q, H k, cl_uint dim, const size_t *off, const size_t *gws,
                              const size_t *lws, cl_uint nw, const H *wl, H *ev);
cl_int clEnqueueNDRangeKernel(H q, H k, cl_uint dim, const size_t *off, const size_t *gws,
                              const size_t *lws, cl_uint nw, const H *wl, H *ev) {
  init();
  H local = 0;
  H *out = ev ? ev : &local;
  uint64_t t0 = now();
  cl_int r = ((cl_int(*)(H, H, cl_uint, const size_t *, const size_t *, const size_t *, cl_uint,
                         const H *, H *))dlsym(real, "clEnqueueNDRangeKernel"))(
      q, k, dim, off, gws, lws, nw, wl, out);
  stats->h[8] += now() - t0;
  stats->h[9]++;
  if (r == 0 && *out && np < MAX_PENDING) {
    int slot = MAX_SLOTS - 1;
    for (int i = 0; i < nk; i++)
      if (kmap[i].kernel == k) { slot = kmap[i].slot; break; }
    if (ev) ((cl_int(*)(H))dlsym(real, "clRetainEvent"))(*ev);
    pending[np].event = *out;
    pending[np].slot = slot;
    np++;
  } else if (!ev && local) {
    ((cl_int(*)(H))dlsym(real, "clReleaseEvent"))(local);
  }
  return r;
}

// Reads the profiling events of finished launches into the counters.
static void drain(void) {
  cl_int (*info)(H, cl_uint, size_t, void *, size_t *) = dlsym(real, "clGetEventProfilingInfo");
  cl_int (*rel)(H) = dlsym(real, "clReleaseEvent");
  for (int i = 0; i < np; i++) {
    cl_ulong s = 0, e = 0;
    info(pending[i].event, CL_PROFILING_COMMAND_START, 8, &s, 0);
    info(pending[i].event, CL_PROFILING_COMMAND_END, 8, &e, 0);
    stats->slots[pending[i].slot].count++;
    stats->slots[pending[i].slot].ns += e - s;
    rel(pending[i].event);
  }
  np = 0;
}

cl_int clFinish(H q);
cl_int clFinish(H q) {
  init();
  uint64_t t0 = now();
  cl_int r = ((cl_int(*)(H))dlsym(real, "clFinish"))(q);
  stats->h[10] += now() - t0;
  stats->h[11]++;
  drain();
  return r;
}

// With NXCL_PROF_FINISH set, a flush waits for the queue. That gives the time split of
// a run that ends in clFinish.
cl_int clFlush(H q);
cl_int clFlush(H q) {
  init();
  if (getenv("NXCL_PROF_FINISH")) return clFinish(q);
  return ((cl_int(*)(H))dlsym(real, "clFlush"))(q);
}

H clCreateBuffer(H ctx, cl_mem_flags flags, size_t size, void *host, cl_int *err);
H clCreateBuffer(H ctx, cl_mem_flags flags, size_t size, void *host, cl_int *err) {
  init();
  uint64_t t = now();
  H b = ((H(*)(H, cl_mem_flags, size_t, void *, cl_int *))dlsym(real, "clCreateBuffer"))(
      ctx, flags, size, host, err);
  uint64_t d = now() - t;
  int i = (flags & CL_MEM_COPY_HOST_PTR) ? 0 : 4;
  stats->h[i] += d;
  stats->h[i + 1]++;
  return b;
}

cl_int clEnqueueReadBuffer(H q, H buf, cl_uint block, size_t off, size_t size, void *ptr,
                           cl_uint nw, const H *wl, H *ev);
cl_int clEnqueueReadBuffer(H q, H buf, cl_uint block, size_t off, size_t size, void *ptr,
                           cl_uint nw, const H *wl, H *ev) {
  init();
  uint64_t t = now();
  cl_int r = ((cl_int(*)(H, H, cl_uint, size_t, size_t, void *, cl_uint, const H *, H *))dlsym(
      real, "clEnqueueReadBuffer"))(q, buf, block, off, size, ptr, nw, wl, ev);
  stats->h[2] += now() - t;
  stats->h[3]++;
  if (block) drain();
  return r;
}

cl_int clBuildProgram(H p, cl_uint n, const H *devs, const char *opts, void *cb, void *ud);
cl_int clBuildProgram(H p, cl_uint n, const H *devs, const char *opts, void *cb, void *ud) {
  init();
  uint64_t t = now();
  cl_int r = ((cl_int(*)(H, cl_uint, const H *, const char *, void *, void *))dlsym(
      real, "clBuildProgram"))(p, n, devs, opts, cb, ud);
  stats->h[6] += now() - t;
  stats->h[7]++;
  return r;
}
