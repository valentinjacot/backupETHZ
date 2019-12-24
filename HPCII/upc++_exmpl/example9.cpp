/**********************************************************************
 * Code: UPC++ - Conjoining Futures with when_all
 * Author: Sergio Martin
 * ETH Zuerich - HPCSE II (Spring 2019)
 **********************************************************************/

#include <stdio.h>
#include <upcxx/upcxx.hpp>

bool finished;
int rankId;

int main(int argc, char* argv[])
{
  upcxx::init();
  rankId    = upcxx::rank_me();
  int rankCount = upcxx::rank_n();

  finished = false;
  upcxx::barrier();

  // Creating a new future to hold all RPC's.
  upcxx::future<> futs = upcxx::make_future();

  if (rankId == 0)
    for (int i = 1; i < rankCount; i++)
    {
     auto fut = upcxx::rpc(i, [](int par){
       printf("Rank %d: Received RPC with Parameter: %d\n", rankId, par);
       finished = true;
      }, i*rankCount);
     futs = upcxx::when_all(futs, fut);
     // We join all futures in a single future to wait for.
    }

  if (rankId > 0) while (!finished) upcxx::progress();

  if (rankId == 0) printf("Not all finished yet.\n");
	if (rankId == 0) futs.wait();
	if (rankId == 0) printf("All finished now.\n");

  upcxx::finalize();
  return 0;
}


