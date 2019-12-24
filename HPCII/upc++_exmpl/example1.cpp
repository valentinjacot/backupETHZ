/**********************************************************************
 * Code: UPC++ Hello World!
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

  printf("Hello, I am rank %d of %d\n", rankId, rankCount);

  upcxx::finalize();
  return 0;
}


