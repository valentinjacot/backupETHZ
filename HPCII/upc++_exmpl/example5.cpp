/**********************************************************************
 * Code: UPC++ Accessing a single shared memory partition through
 *       sharing it's global pointer with a broadcast.
 * Author: Sergio Martin
 * ETH Zuerich - HPCSE II (Spring 2019)
 **********************************************************************/

#include <stdio.h>
#include <upcxx/upcxx.hpp>

int main(int argc, char* argv[])
{
  upcxx::init();
  int rankId    = upcxx::rank_me();
  int rankCount = upcxx::rank_n();

  upcxx::global_ptr<int> gptr;
  if (rankId == 0) gptr = upcxx::new_array<int>(rankCount);

  upcxx::broadcast(&gptr, 1, 0).wait();

  auto future = upcxx::rput(&rankId, gptr + rankId, 1);

  printf("Rank %d - Updating the global allocation.\n", rankId);

  future.wait();

  upcxx::barrier();

  if (rankId == 0)
  {
   int* lptr = gptr.local();
   printf("{");
   for (int i = 0; i < rankCount; i++)
    printf("%d,", lptr[i]);
   printf("}\n");
  }

  upcxx::finalize();
  return 0;
}


