/**********************************************************************
 * Code: UPC++ - RPC's with return values.
 * Author: Sergio Martin
 * ETH Zuerich - HPCSE II (Spring 2019)
 **********************************************************************/


#include <stdio.h>
#include <upcxx/upcxx.hpp>

int calculateSquare(int x) { return x*x; }

int main(int argc, char* argv[])
{
  upcxx::init();
  int rankId    = upcxx::rank_me();
  int rankCount = upcxx::rank_n();

  if (rankId == 0)
   for (int i = 1; i < rankCount; i++)
    printf("Value: %d - Square %d\n", i, upcxx::rpc(i, calculateSquare, i).wait());

  upcxx::barrier();
  upcxx::progress();
  upcxx::finalize();
  return 0;
}


