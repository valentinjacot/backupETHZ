package patterns.observer.cycle;

import java.awt.Color;

public class ColorModel extends Observable {
	private Color color;

	public ColorModel(Color color) {
		this.color = color;
	}

	public Color getColor() {
		return color;
	}

	public void setColor(Color color) {
		System.out.println("ColorModel.setColorChanged");
		this.color = color;
		notifyObservers(color);
	}
}

/*
		if (!color.equals(this.color)) {
			this.color = color;
			notifyObservers(color);
		}

 */