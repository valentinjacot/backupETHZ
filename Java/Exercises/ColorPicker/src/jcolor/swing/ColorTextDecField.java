package jcolor.swing;
import java.awt.Color;
import java.awt.event.FocusEvent;
import java.awt.event.FocusListener;
import javax.swing.JTextField;
import javax.swing.event.DocumentEvent;
import javax.swing.event.DocumentListener;
import jcolor.ColorChannel;
import jcolor.ColorListener;
import jcolor.ColorModel;
class ColorTextDecField extends JTextField
implements DocumentListener, FocusListener, ColorListener {
	private ColorModel model;
	private ColorChannel channel;
	ColorTextDecField(ColorModel model, ColorChannel channel) {
		super("", 5);
		this.model = model;
		this.channel = channel;
		getDocument().addDocumentListener(this);
		addFocusListener(this);
		model.addColorListener(this);
	}
	// DocumentListener implementation
	@Override
	public void insertUpdate(DocumentEvent e) {
		textChangeNotification();
	}
	@Override
	public void removeUpdate(DocumentEvent e) {
		textChangeNotification();
	}
	@Override
	public void changedUpdate(DocumentEvent e) {
		textChangeNotification();
	}
	private void textChangeNotification() {
		try {
			int value = Integer.parseInt(getText());
			if (value >= 0 && value < 256) {
				model.setColor(channel.modifiedColor(model.getColor(), value));
			}
		} catch (NumberFormatException e) {
			// do nothing, i.e. keep old color value
			// but characters remain in the text field
			// model.setColor(c) does not help as color does not change
		} catch (Exception x) {
			x.printStackTrace();
		}
	}
	//FocusListener implementation
	@Override
	public void focusGained(FocusEvent e) {
	}
	@Override
	public void focusLost(FocusEvent e) {
		try {
			ColorTextDecField.super.setText("" + Integer.parseInt(getText()));
		} catch (Exception ex) {
			super.setText("" + channel.getValue(model.getColor()));
		}
	}
	//ColorListener implementation
	@Override
	public void colorValueChanged(Color color) {
		setText("" + channel.getValue(color));
	}
}