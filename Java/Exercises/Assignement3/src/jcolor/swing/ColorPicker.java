package jcolor.swing;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Container;
import java.awt.GridLayout;
import java.awt.Scrollbar;

import javax.swing.JButton;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.JScrollBar;
import javax.swing.JTextField;
import javax.swing.SwingUtilities;
import jcolor.ColorModel;
import jcolor.ColorChannel;
class ColorPicker extends JFrame {
	public static void main(String[] args) {
		SwingUtilities.invokeLater(new Runnable() {
			public void run() {
				JFrame f = new ColorPicker();
				f.pack();
				f.setVisible(true);
			}
		});
	}
	private ColorModel model = new ColorModel();
	// Color model;
	ColorPicker() {
		super("Color Picker");
		setBackground(Color.lightGray);
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		Container c = getContentPane();
		c.setLayout(new BorderLayout());
		final ColorScrollBar sb_r = new ColorScrollBar(model, ColorChannel.RED,Scrollbar.HORIZONTAL,0);
		final ColorScrollBar sb_b = new ColorScrollBar(model, ColorChannel.BLUE,Scrollbar.HORIZONTAL,0);
		final ColorScrollBar sb_g = new ColorScrollBar(model, ColorChannel.GREEN,Scrollbar.HORIZONTAL,0);
		final JButton button1 = new JButton("Button1");

		Color r = Color.red;
		Color b = Color.blue;
		Color g = Color.green;
		JTextField text_r1 = new JTextField();
		JTextField text_r2 = new JTextField();text_r2.setEditable(false);
		JTextField text_b1 = new JTextField();
		JTextField text_b2 = new JTextField();text_b2.setEditable(false);
		JTextField text_g1 = new JTextField();
		JTextField text_g2 = new JTextField();text_g2.setEditable(false);

		JPanel upperLeft = new JPanel();
		upperLeft.setLayout(new GridLayout(3, 1));
		upperLeft.add(sb_r);upperLeft.add(text_r1);upperLeft.add(text_r2);
		upperLeft.add(sb_b);upperLeft.add(text_b1);upperLeft.add(text_b2);
		upperLeft.add(sb_g);upperLeft.add(text_g1);upperLeft.add(text_g2);

		JPanel bottom = new JPanel();
		bottom.setLayout(new GridLayout(3, 1));

		bottom.add(new ColorField(model));
		

		c.add(upperLeft, BorderLayout.NORTH);
		c.add(bottom, BorderLayout.SOUTH);

		this.setDefaultCloseOperation(EXIT_ON_CLOSE);
	}
}
