//member access
//_______________________________________________________________

class C {
	//default private !!
	int a; //private --> can be accessed only from within the class
public:// can be accessed only by derived class
	int b;
protected:// can be accessed by anyone
int c;
}
//MAIN DIFFERENCE BETWEEN CLASS & STRUCT:
//sturct members are public by default
