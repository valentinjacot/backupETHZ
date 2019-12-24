//STRUCT & CLASSES
//---------------------------

struct myStruct {
	int some_int;
	double some_double;
	void incr(){some_int++;} //function
};
class myClass {
	int I;
	myStruct S; //contains a struct
};

class myClass {
	public:
	myClass() {I=9;}; //default contructor ------>  myClass m; 
	myClass(int I_) : I(I_) {} // any constructor  ----> I(I_) is like I = I_;
	myClass(const myClass & other)//copy constructor
		:I(other.I){} //spectial sytax
	int I;
};

class myClass {
	int operator*(const myClass & other) const {
		return 5*I + other.I;
	}
	
	static int static_I; //can be access form everywhere (main, whatever)
}

struct T {
	int some_int=9;
	T & operator=(const T & other) {
	this -> i = other.i; //not required (implicit)
						//this->i <=> *this
	return *this;}
};



