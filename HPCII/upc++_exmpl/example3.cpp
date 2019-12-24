/**********************************************************************
 * Code: UPC++ Blocking Broadcast Example
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

  int number = 0;
  if (rankId == 0) number = 500;

  // Notice the wait at the end of the broadcast!
  upcxx::broadcast(&number, 1 /*Count*/, 0 /*Root*/).wait();

  printf("Rank %d: Number is %d.\n", rankId, number);

  upcxx::finalize();
  return 0;
}


