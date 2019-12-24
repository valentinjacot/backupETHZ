package patterns.decorator;

import java.awt.Graphics;

public class RedDecorator implements Phone {
	private final Phone inner;

	public RedDecorator(Phone inner) {
		this.inner = inner;
	}

	@Override
	public void dial(String nr) {
		inner.dial(nr);
	}

	@Override
	public void draw(Graphics g) {
		inner.draw(g);
		System.out.println("Drawing red decorator.");
	}

}
