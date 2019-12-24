package jcolor;
import java.awt.Color;
public enum ColorChannel {
	RED(Color.RED) {
		@Override
		public int getValue(Color color) {
			return color.getRed();
		}
		@Override
		public Color modifiedColor(Color color, int value) {
			return new Color(value, color.getGreen(), color.getBlue());
		}
	},
	GREEN(Color.GREEN) {
		@Override
		public int getValue(Color color) {
			return color.getGreen();
		}
		@Override
		public Color modifiedColor(Color color, int value) {
			return new Color(color.getRed(), value, color.getBlue());
		}
	},
	BLUE(Color.BLUE) {
		@Override
		public int getValue(Color color) {
			return color.getBlue();
		}
		@Override
		public Color modifiedColor(Color color, int value) {
			return new Color(color.getRed(), color.getGreen(), value);
		}
	};
	ColorChannel(Color color) { this.color = color; }
	private Color color;
	public Color getColor() { return color; }
	public abstract int getValue(Color color);
	public abstract Color modifiedColor(Color color, int value);
}