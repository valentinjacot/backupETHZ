/**********************************************************************
 * Code: UPC++ - Accessing multi-partitioned global shared memory with
 *       a distributed object.
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

  auto myPartition = upcxx::new_<double>(rankId);
  upcxx::dist_object<upcxx::global_ptr<double>> partitions(myPartition);

  *myPartition.local() = sqrt((double)rankId);

  upcxx::barrier();

  if (rankId == 0) for (int i = 0; i < rankCount; i++)
  {
  	auto partition = partitions.fetch(i).wait();
  	double val = upcxx::rget(partition).wait();
  	printf("SquareRoot(%f) = %f\n", (double)i, val);
  }

  upcxx::finalize();
  return 0;
}


