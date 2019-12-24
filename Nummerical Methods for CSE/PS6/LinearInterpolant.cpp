///questions are: why is it not working
///I don't understand the ML



#include <iostream>
#include <utility>
#include <vector>



class LinearInterpolant {
public:
LinearInterpolant( const std::vector<double> t_i, const std::vector<double> y_i ) {
	n= t_i.size();
	for (size_t i= 0, i<n, i++){
		val_vect = new std::pair<double,double>[n];	
		val_vect[i].first = t_i[i];
		val_vect[i].second = y_i[i];
		}
	
};

double operator() (double t) {
	b= new b[n];
	// b 0
	if (t>=val_vect[0].first && t<val_vect[1].first){b[0]=1-((t-val_vect[0])/(val_vect[1]-val_vect[0]));}
	else{b[0]=0;};
	//b n-1
	if (t>=val_vect[n-2].first && t<val_vect[n-1].first){b[n-1]=1-((val_vect[n-1]-t)/(val_vect[n-1]-val_vect[n-2]));}
	else{b[n-1]=0;};
	// b j
	for (int j=1; j<n-1; j++){
		if (t>=val_vect[j-1].first && t<val_vect[j].first){b[j]=1-((val_vect[1]-t)/(val_vect[j]-val_vect[j-1]));}
		else if(t>=val_vect[j].first && t<val_vect[j+1].first){b[j]=1-((t-val_vect[j])/(val_vect[j+1]-val_vect[j]));}
	else{b[j]=0;};
} return b;
};
private:
	std::pair<double,int>* val_vect;
	double* b;
	size_t n;
};


int main(void) {
    // Test the class with the basis with nodes (-1,1,2,4) and interpolant with values (-1,2,3,4) at said nodes.
    LinearInterpolant I = LinearInterpolant({{1,2},{2,3},{4,4},{-1,-1}});
    // Output value at the specified point
    std::cout << I(-2) << " " << I(-1) << " " << I(1) << " " << I(2) << " " << I(4) << " " << I(5) << std::endl;
    std::cout << I(0.5) << " " << I(1) << " " << I(1.5) << " " << I(2.1) << " " << I(3) << " " << I(3.1) << " " << I(4) << std::endl;
}



