/**********************************************************************
 * Code: UPC++ Example of Remote Procedure Calls
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
      upcxx::rpc(i, [](int par){
       printf("Rank %d: Received RPC with Parameter: %d\n", rankId, par);
       finished = true;
      }, i*rankCount).wait();
  // In this case, we wait for each RPC to finish before we continue with another.

  if (rankId > 0) while (!finished) upcxx::progress();

  upcxx::finalize();
  return 0;
}


