package jcolor.swing;
import javax.swing.JScrollBar;
import jcolor.ColorChannel;
import jcolor.ColorModel;
class ColorScrollBar extends JScrollBar{
	ColorScrollBar(ColorModel model, ColorChannel channel, 
			int orientation, int val) {
		super(orientation, val, 0, 0, 255);
		setBackground(channel.getColor());
		addAdjustmentListener(e -> model.setColor(
				channel.modifiedColor(model.getColor(), getValue())));
		model.addColorListener(c -> setValue(channel.getValue(c)));
	}
}