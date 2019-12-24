class Trafficlight {
public: // access declaration
	enum light {green, orange, red}; //type member
	// Trafficlight(); //default constructor
	Trafficlight(light=red); //constructor
	~Trafficlight(); //destructor
	
	light state() const; // const -> because it doesn't change anything
	void set_state(light); //void because it doesn't return anything, but not 'const' 'cause it changes smth
private: //this is hidden
	light state_; // data member
}

//usage :
Trafficlight
x(trafficlight::green);
Trafficlight::light l;

l=x.state();
l=Trafficlight::green;


// constructor
int& x;
int const y;
A(int& r, int s)
	:x(r),
	y(s)
	{
	}
//const & volatile
//mutable
// friend
