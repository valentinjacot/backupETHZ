
public class Complex{
	private double x, y;
	public Complex(double x_, double y_){
		x=x_; y=y_;
	}
	public Complex(Complex c) { //copy constructor 
		x=c.x; y=c.y;
	}
	public double getReal(){return x;}
	public double getImg(){return y;}
	public void setReal(double x_){this.x=x_;}
	public void setImg(double y_){this.y=y_;}
	//static methods (returns a new object, don't change the actual instance)
	public static Complex sum(Complex a, int b){
		return new Complex(a.getReal()+b, a.getImg());
	}
	public static Complex sum(Complex a, Complex b){
		return new Complex(a.getReal()+b.getReal(), a.getImg()+b.getImg());
	}
	public static Complex multiply(Complex a, Complex b){
		return new Complex(a.getReal()*b.getReal() - 
				a.getImg()*b.getImg(),a.getReal()*b.getImg() + a.getImg()*b.getReal());
	}
	public static Complex asComplex(int a) {
		return new Complex(a,0.0);
	}
	public static Complex asComplex(double a) {
		return new Complex(a,0.0);
	}
	public double norm(){
		return Math.sqrt(x*x+y*y);
	}
	public Complex square(){
		return new Complex (x*x - y*y, 2*x*y);
	}
	public double abs(){
		return Math.sqrt(this.norm());
	}
	public void AddUp(Complex c){
		x+=c.x;y+=c.y;
	}
	
	public void print(){System.out.println(x + " + " + y + "i");}
	public static final Complex ZERO = new Complex(0,0);
	public static final Complex ONE = new Complex(1,0);
	public static final Complex I = new Complex(0,1);
	
}

