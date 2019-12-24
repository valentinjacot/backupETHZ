package jcolor.swing;
import javax.swing.JTextField;
import jcolor.ColorChannel;
import jcolor.ColorModel;
class ColorTextHexField extends JTextField {
	ColorTextHexField(ColorModel model, ColorChannel channel) {
		super("", 3);
		setEditable(false);
		model.addColorListener(
				c -> setText(Integer.toHexString(channel.getValue(c)))
				);
	}
}