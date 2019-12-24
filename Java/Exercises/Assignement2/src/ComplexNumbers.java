import java.lang.Object;
import javax.swing.JFrame;
import javax.swing.JPanel;

public class ComplexNumbers {

	/**
	 * @param args
	 */
	public static void main(String[] args) {
		
		Complex a = new Complex(1,2);
		Complex a1= new Complex(0,1);
		Complex b = new Complex(3,4);
		Complex c = Complex.sum(a, b);
		Complex d = Complex.multiply(a, b);
		a.print();
		b.print();
		c.print();
		d.print();
		System.out.println(d.norm());
		Complex a2 = a1.square();
		a2.print();
		int i =2;
		Complex a3 = Complex.asComplex(i);
		System.out.println(a3);
		a3.print();
		a1.print();
//		System.out.println(a1.norm());
//		System.out.println(b.abs());
		Complex zer = new Complex(2,0);
		System.out.println(zer.abs());
		zer.print();
		Complex new_complex = new Complex(c);
		new_complex.print();
		Complex nc_2 = new_complex.square();
		nc_2.print();
		Complex n = Complex.ONE;
		n.print();
		
	}
}