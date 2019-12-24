package graphics;

public class Complex {
	private final double real, img;
	public Complex(double r, double i) {
		real = r;
		img = i;
	}
	public Complex(Complex c) {
		real = c.real;
		img = c.img;
	}
	public static Complex sum(Complex c1, Complex c2) {
		return new Complex(c1.real + c2.real, c1.img + c2.img);
	}
	public static Complex multiply(Complex c1, Complex c2) {
		return new Complex(c1.real * c2.real - c1.img * c2.img, c1.real * c2.img + c1.img * c2.real);
	}
//	public void addTo(Complex c) {
//		this.real += c.real;
//		this.img += c.img;
//	}
//	public void multiplyWith(Complex c) {
//		double tempReal=this.real;
//		double tempImg=this.img;
//		this.real = tempReal * c.real - tempImg * c.img;
//		this.img = tempReal * c.img + tempImg * c.real;
//	}
	public double getReal() {
		return this.real;
	}
	public double getImg() {
		return this.img;
	}
//	public void setReal(double r) {
//		this.real = r;
//	}
//	public void setImg(double i) {
//		this.img = i;
//	}
	public void print() {
		System.out.println(real + " + " + img + "i");
	}
	public String toString() {
		return real + " + " + img + "i \n";
	}
	// static?
	public Complex square() {
		return new Complex(this.real * this.real - this.img * this.img, 2 * this.real * this.img);
		//return new Complex(multiply(this,this));
	}
	public double norm() {
		return Math.sqrt(real * real + img * img);
	}
	public static Complex asComplex(double i) {
		return new Complex(i,0);
	}
	public static Complex asComplex(int i) {
		return new Complex(i,0);
	}
	public double abs() {
		return Math.sqrt(this.norm());
	}
	public static final Complex ZERO = new Complex (0,0);
	public static final Complex ONE= new Complex (1,0);
	public static final Complex I = new Complex (0,1);
}
