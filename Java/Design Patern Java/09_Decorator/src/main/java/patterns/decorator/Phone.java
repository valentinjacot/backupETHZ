package patterns.decorator;

import java.awt.Graphics;

public interface Phone {
	public void dial(String nr);
	public void draw(Graphics g);
}

class Mobile implements Phone {

	@Override
	public void dial(String nr) {
		System.out.println("Calling number " + nr);
	}

	@Override
	public void draw(Graphics g) {
		System.out.println("Drawing phone.");
	}

}
