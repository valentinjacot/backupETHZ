/**********************************************************************
 * Code: UPC++ - Composing Futures with future.then();
 * Author: Sergio Martin
 * ETH Zuerich - HPCSE II (Spring 2019)
 **********************************************************************/

#include <stdio.h>
#include <upcxx/upcxx.hpp>

bool finished;
int rankId;
int rankCount;

int main(int argc, char* argv[])
{
  upcxx::init();
  rankId    = upcxx::rank_me();
  rankCount = upcxx::rank_n();

  finished = false;
  upcxx::barrier();

  if (rankId == 0)
  {
   upcxx::future<> futs = upcxx::make_future();
   for (int i = 1; i < rankCount; i++)
   {
    auto f1 = upcxx::rpc(i, [](int par){
     printf("Rank %d: Received RPC with Parameter: %d\n", rankId, par);
     finished = true;
    }, i*rankCount);

    // Defining callback: report the return of a rank after it printed its message
    f1.then([i](){printf("Rank 0: Rank %d Came back.\n", i);});

    // Conjoing all futures into one.
    futs = upcxx::when_all(futs,f1);
   }
   // Wait for futures.
   futs.wait();
  }
  if (rankId > 0) while (!finished) upcxx::progress();

  upcxx::finalize();
  return 0;
}


