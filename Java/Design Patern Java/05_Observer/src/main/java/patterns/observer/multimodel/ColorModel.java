package patterns.observer.multimodel;

import java.awt.Color;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.Map;

public class ColorModel {
	private int red, green, blue;

	public enum ColorChannel {
		RED, GREEN, BLUE
	}

	public ColorModel(Color c) {
		this.red = c.getRed();
		this.green = c.getGreen();
		this.blue = c.getBlue();
	}

	private final Map<ColorListener, EnumSet<ColorChannel>> listeners = new HashMap<>();

	// register a color listener for a set of channels, i.e. a subset of {RED,GREEN,BLUE}
	public void addColorListener(ColorListener l, EnumSet<ColorChannel> set) {
		listeners.put(l, set);
	}

	public void removeColorListener(ColorListener l) {
		listeners.remove(l);
	}

	public void setColor(Color c) {
		setRed(c.getRed());
		setGreen(c.getGreen());
		setBlue(c.getBlue());
	}

	public Color getColor() {
		return new Color(red, green, blue);
	}

	public void setRed(int red) {
		this.red = red;
		notifyListeners(ColorChannel.RED);
	}

	public void setGreen(int green) {
		this.green = green;
		notifyListeners(ColorChannel.GREEN);
	}

	public void setBlue(int blue) {
		this.blue = blue;
		notifyListeners(ColorChannel.BLUE);
	}

	private void notifyListeners(ColorChannel channel) {
		Color color = getColor();
		for (ColorListener l : new ArrayList<>(listeners.keySet())) {
			if (listeners.get(l).contains(channel))
				l.colorValueChanged(color);
		}
	}
	
}

/*

	public void setColor(Color c) {
		EnumSet<ColorChannel> s = EnumSet.noneOf(ColorChannel.class);
		if (red != c.getRed()) s.add(ColorChannel.RED);
		if (green != c.getGreen()) s.add(ColorChannel.GREEN);
		if (blue != c.getBlue()) s.add(ColorChannel.BLUE);
		red = c.getRed();
		green = c.getGreen();
		blue = c.getBlue();
		notifyListeners(s);
	}

	private void notifyListeners(EnumSet<ColorChannel> channels) {
		Color color = getColor();
		for (ColorListener l : new ArrayList<>(listeners.keySet())) {
			if (!Collections.disjoint(listeners.get(l), channels))
				l.colorValueChanged(color);
		}
	}
	
	public void setRed(int red) {
		this.red = red;
		notifyListeners(EnumSet.of(ColorChannel.RED));
	}

	public void setGreen(int green) {
		this.green = green;
		notifyListeners(EnumSet.of(ColorChannel.GREEN));
	}

	public void setBlue(int blue) {
		this.blue = blue;
		notifyListeners(EnumSet.of(ColorChannel.BLUE));
	}


*/
