/**********************************************************************
 * Code: UPC++ Nested RPCs
 * Author: Sergio Martin
 * ETH Zuerich - HPCSE II (Spring 2019)
 **********************************************************************/

#include <stdio.h>
#include <upcxx/upcxx.hpp>

bool finished;
int responses;
int rankId;
int rankCount;

int main(int argc, char* argv[])
{
  upcxx::init();
  rankId    = upcxx::rank_me();
  rankCount = upcxx::rank_n();

  finished = false;
  responses = 0;
  upcxx::barrier();

  if (rankId == 0)
   for (int i = 1; i < rankCount; i++)
    upcxx::rpc_ff(i, [](int par){
     printf("Rank %d: Received RPC with Parameter: %d\n", rankId, par);
     finished = true;
     upcxx::rpc_ff(0, []()
     {
      responses++;
      printf("Rank 0: Received responses: %d\n", responses);
      if (responses == rankCount-1) finished = true;
     });
    }, i*rankCount);

  if (rankId > 0) while (!finished) upcxx::progress();

  upcxx::finalize();
  return 0;
}


