/**********************************************************************
 * Code: UPC++ Quiescence - Using Fire and Forget RPCs
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

  if (rankId == 0)
    for (int i = 1; i < rankCount; i++)
      upcxx::rpc_ff(i, [](int par){
       printf("Rank %d: Received RPC with Parameter: %d\n", rankId, par);
       finished = true;
      }, i*rankCount);

  // In this case, we dont wait for each RPC to finish.
  // We can just "fire-and-forget", by using rpc_ff.

  if (rankId > 0) while (!finished) upcxx::progress();

  upcxx::finalize();
  return 0;
}


