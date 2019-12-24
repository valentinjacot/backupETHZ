int *a; // ptr to int
int * b = new int; // ptr to allocated int
a=b; // a points to the same as b
*a=9; // value at the memory pointed by a is 9
std::cout<< *b; // return 9
int c;
a =&c;// a point to the address of c
delete b;

// --> * (unary) return value of pointer (dereference)
// --> & (unary) return address of variable (dereference)
//new and delete allocate and free some space
double i=9; // <=>doublee i= (double) 9.
// keyword: cast/casting


