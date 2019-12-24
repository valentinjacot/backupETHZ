#include <atomic>
#include <chrono>
#include <cstdio>
#include <omp.h>
#include <random>
#include <thread>

const int MAX_T = 128;  // Maximum number of threads. (*)


class ALock {
private:
    // TODO: Question 3a: Member variables.
    std::atomic<unsigned> tail;  // (*)
    volatile bool flag[MAX_T];
    int slot[MAX_T];

    // (*) Not really necessary, but we use an `unsigned int` as a `tail`, and
    // a power of two `MAX_T`. This way, if we have more than 4*10^9 calls to
    // the lock, `tail` will cycle back to 0 and everything will work fine.
    // Without `MAX_T` being a power of two, `tail % MAX_T` would not be
    // continuous during the jump, while `unsigned int` must be used because
    // `signed int` overflow is undefined behavior.
    // Also, with `MAX_T == 128`, the (very) expensive modulo operation is
    // replaced by a very cheap AND operation!

public:
    ALock() : tail{0}, flag{}, slot{} {
        // TODO: Question 3a: Initial values.
        flag[0] = true;
    }

    void lock(int tid) {
        // TODO: Question 3a
        int s = tail.fetch_add(1) % MAX_T;  // tail++
        slot[tid] = s;

        // Wait until my flag gets set to true.
        while (!flag[s])
            continue;
    }

    void unlock(int tid) {
        // TODO: Question 3a

        // I release the lock and give it to the next thread in the queue
        // (or, if no-one is in the queue, whoever calls `lock` first).
        flag[slot[tid]] = false;
        flag[(slot[tid] + 1) % MAX_T] = true;
    }
};


/*
 * Return time in seconds from the program start.
 */
double get_time() {
    static auto start = std::chrono::steady_clock::now();
    auto now = std::chrono::steady_clock::now();
    std::chrono::duration<double> diff = now - start;
    return diff.count();
}


/*
 * Print the thread ID and the current time.
 */
void log(int tid, const char *info) {
    // TODO: Question 3a: Print the event in the format:
    //  tid     info    time_since_start
    //
    // Note: Be careful with a potential race condition here.

    // Here I assume that printf is thread-safe.
    // https://stackoverflow.com/a/41131253
    printf("%2d %8s %lf\n", tid, info, get_time());
}


/*
 * Sleep for `ms` milliseconds.
 */
void suspend(int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}


void emulate(int factor) {
    ALock lock;  // For the for loop.

    ALock lock_time;  // For the time variables.
    double total_time = 0.0;
    double total_wait = 0.0;

#pragma omp parallel
    {
        // Begin parallel region.
        int tid = omp_get_thread_num();  // Thread ID.
        // Initialize the generator with a unique seed.
        std::mt19937 generator(tid);
        std::uniform_int_distribution<int> Winside(50 / factor, 200 / factor);
        std::uniform_int_distribution<int> Woutside(50, 200);

        double start_time = 0;
        double wait_time = 0;

        // TODO: Question 3b: Repeat multiple times
        //      - log(tid, "BEFORE")
        //      - lock
        //      - log(tid, "INSIDE")
        //      - Winside
        //      - unlock
        //      - log(tid, "AFTER")
        //      - Woutside
        //
        // NOTE: Make sure that:
        //      - there is no race condition in the random number generator
        //      - each thread computes different random numbers
        for (int i = 0; i < 5; ++i) {
            log(tid, "BEFORE");

            double before_lock = get_time();
            lock.lock(tid);
            wait_time += get_time() - before_lock;

            log(tid, "INSIDE");
            suspend(Winside(generator));

            lock.unlock(tid);

            log(tid, "AFTER");
            suspend(Woutside(generator));
        }

        // A different lock such that it doesn't interfere with the one inside
        // the for loop. Here we don't count the last wait (for everyone to
        // finish), but that doesn't change the result much.
        lock_time.lock(tid);
        double time = get_time() - start_time;
        total_time += time;
        total_wait += wait_time;
        fprintf(stderr, "Thread #%d waiting for %lf/%lfs.\n",
                tid, wait_time, time);
        lock_time.unlock(tid);

        // End parallel region.
    }

    fprintf(stderr,
            "Waiting time for <W_inside>/<W_outside> = 1/%d is %.2lf%%\n",
            factor, 100. * total_wait / total_time);
}


/*
 * Test that a lock works properly by executing some calculations.
 */
void test_alock() {
    const int N = 1000000, A[2] = {2, 3};
    int result = 0, curr = 0;
    ALock lock;

#pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        int tid = omp_get_thread_num();  // Thread ID.
        lock.lock(tid);

        // Something not as trivial as a single ++x.
        result += A[curr = 1 - curr];

        lock.unlock(tid);
    }

    int expected = (N / 2) * A[0] + (N - N / 2) * A[1];
    if (expected == result) {
        fprintf(stderr, "Test OK!\n");
    } else {
        fprintf(stderr, "Test NOT OK: %d != %d\n", result, expected);
        exit(1);
    }
}


int main() {
    test_alock();

    // TODO: Question 3b:
    emulate(1);    // Same Winside and Woutside on average.
    emulate(10);   // 10 times shorter Winside than Woutside on average.

    return 0;
}
