package jcolor.swing;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Container;
import java.awt.FlowLayout;
import java.awt.GridLayout;
import java.awt.Scrollbar;
import javax.swing.JFrame;
import javax.swing.JMenu;
import javax.swing.JMenuBar;
import javax.swing.JMenuItem;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;
import jcolor.ColorChannel;
import jcolor.ColorModel;
public class ColorApplication extends JFrame {
	public static void main(String[] args) {
		SwingUtilities.invokeLater(() -> {
			JFrame frame = new ColorApplication();
			frame.pack();
			frame.setVisible(true);
		});
	}
	private ColorModel model = new ColorModel();
	ColorApplication(){
		setTitle("Color Picker Swing");
		setBackground(Color.lightGray);
		setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		Container c = getContentPane();
		c.setLayout(new BorderLayout());
		JPanel top = new JPanel(new GridLayout(1, 2, 5, 5));
		c.add(top, BorderLayout.NORTH);
		JPanel bottom = new JPanel(new FlowLayout());
		c.add(bottom, BorderLayout.CENTER);
		JPanel p;
		// Scrollbar panelk
		p = new JPanel(new GridLayout(3,1,3,3));
		top.add(p);
		p.add(new ColorScrollBar(model, ColorChannel.RED,
				Scrollbar.HORIZONTAL, 0));
		p.add(new ColorScrollBar(model, ColorChannel.GREEN,
				Scrollbar.HORIZONTAL, 0));
		p.add(new ColorScrollBar(model, ColorChannel.BLUE,
				Scrollbar.HORIZONTAL, 0));
//		Color Picker (Implementation using the Observer Pattern)
		// Textfield panel
		p = new JPanel(new GridLayout(3, 2));
		top.add(p);
		p.add(new ColorTextDecField(model, ColorChannel.RED));
		p.add(new ColorTextHexField(model, ColorChannel.RED));
		p.add(new ColorTextDecField(model, ColorChannel.GREEN));
		p.add(new ColorTextHexField(model, ColorChannel.GREEN));
		p.add(new ColorTextDecField(model, ColorChannel.BLUE));
		p.add(new ColorTextHexField(model, ColorChannel.BLUE));
		// Color Field
		bottom.add(new ColorField(model));
		// CheckBox panel
		p = new JPanel(new GridLayout(0,1));
		bottom.add(p);
		p.add(new ColorRadioButton(model, "red", Color.red));
		p.add(new ColorRadioButton(model, "blue", Color.blue));
		p.add(new ColorRadioButton(model, "green", Color.green));
		p.add(new ColorRadioButton(model, "yellow", Color.yellow));
		p.add(new ColorRadioButton(model, "cyan", Color.cyan));
		p.add(new ColorRadioButton(model, "orange", Color.orange));
		// Button panel
		p = new JPanel(new GridLayout(2, 1, 5, 5));
		bottom.add(p);
		p.add(new ColorButton(model, ColorButton.Type.DARKER, "Darker"));
		p.add(new ColorButton(model, ColorButton.Type.BRIGHTER, "Brighter"));
		JMenuBar bar = new JMenuBar();
		setJMenuBar(bar);
		JMenu file = new JMenu("File");
		bar.add(file);
		JMenuItem exit = new JMenuItem("Exit");
		file.add(exit);
		exit.addActionListener(e -> System.exit(0));
		JMenu attr = new JMenu("Attributes");
		bar.add(attr);
		attr.add(new ColorMenuItem(model, "red", Color.red));
		attr.add(new ColorMenuItem(model, "blue", Color.blue));
		attr.add(new ColorMenuItem(model, "green", Color.green));
		attr.add(new ColorMenuItem(model, "cyan", Color.cyan));
		attr.add(new ColorMenuItem(model, "pink", Color.pink));
		attr.add(new ColorMenuItem(model, "orange", Color.orange));
		attr.add(new ColorMenuItem(model, "magenta",Color.magenta));
		attr.add(new ColorMenuItem(model, "gray", Color.gray));
		attr.add(new ColorMenuItem(model, "black", Color.black));
		model.setColor(Color.black); // update all controls
	}
}