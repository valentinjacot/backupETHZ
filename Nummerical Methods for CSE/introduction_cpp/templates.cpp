//Templates 
//___________________________________________________

//idea: let the compiler write the code for you, juste build a sceleton

template <class whatever>
whatever AXPY (const whatever & X,
				const whatever & Y, double A)
				{return A*X+Y;}
				
template <class T1, class T2> //s.t. can mix T1 & T2
T1 myMin (const T1 &a, const T2 &b){
	return a < b? a: b; //temporary operator
}
int a = myMin(8., 9.); //T1=T2=double
//ect..

//class Template

template <class T>
struct myComplex {
	T Re, Im; //real & im part
	
	myComplex(T Re, T Im) : Re(Re), Im(Im){};
};
myComplex<int> cmplx1(8,9);
myComplex<double> cmplx1(8.4,9.3);




