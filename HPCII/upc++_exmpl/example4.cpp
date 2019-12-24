/**********************************************************************
 * Code: UPC++ Non-Blocking Broadcast Example
 * Author: Sergio Martin
 * ETH Zuerich - HPCSE II (Spring 2019)
 **********************************************************************/

#include <stdio.h>
#include <upcxx/upcxx.hpp>

int main(int argc, char* argv[])
{
 upcxx::init();
 int rankId = upcxx::rank_me();
 int rankCount = upcxx::rank_n();

 int number = 0;
 if (rankId == 0) number = 500;

 // Creating a future as a promise of completion for the broadcast.
 auto future = upcxx::broadcast(&number, 1 /*Count*/, 0 /*Root*/);

 printf("Rank %d: (Before) Number is %d.\n", rankId, number);

 // Do stuff...

 // Now we place the wait after we've done some other stuff
 future.wait();

 printf("Rank %d: (After) Number is %d.\n", rankId, number);

 upcxx::finalize();
 return 0;
}

