package jcolor.swing;
import jcolor.ColorModel;
import jcolor.ColorChannel;
import javax.swing.JTextField;
public class ColorTextHexField extends JTextField{
	ColorTextHexField(ColorModel model,ColorChannel channel){
		super("",3);
		setEditable(false);
		model.addColorListener(
				c -> setText(Integer.toHexString(channel.getValue(c)))
				);
	}
}
