/**********************************************************************
 * Code: UPC++ Barrier Example
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

  printf("Rank %d: Let's do this first.\n", rankId);

  upcxx::barrier();

  printf("Rank %d: Let's do this second.\n", rankId);

  upcxx::finalize();
  return 0;
}


