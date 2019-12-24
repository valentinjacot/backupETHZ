#include <atomic>
#include <chrono>
#include <omp.h>
#include <random>
#include <thread>

const int MAX_T = 100;  // Maximum number of threads.

class ALock {
private:
    // TODO: Question 3a: Member variables.

public:
    ALock() {
        // TODO: Question 3a: Initial values.
    }

    void lock(int tid) {
        // TODO: Question 3a
    }

    void unlock(int tid) {
        // TODO: Question 3a
    }
};


/*
 * Print the thread ID and the current time.
 */
void log(int tid, const char *info) {
    // TODO: Question 3a: Print the event in the format:
    //  tid     info    time_since_start
    //
    // Note: Be careful with a potential race condition here.
}


/*
 * Sleep for `ms` milliseconds.
 */
void suspend(int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}


void emulate() {
    ALock lock;

#pragma omp parallel
    {
        // Begin parallel region.
        int tid = omp_get_thread_num();  // Thread ID.

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

        // End parallel region.
    }
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
    // emulate();

    return 0;
}
