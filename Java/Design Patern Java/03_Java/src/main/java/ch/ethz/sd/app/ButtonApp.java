package ch.ethz.sd.app;
import java.awt.*;
import java.awt.event.*;

@SuppressWarnings("serial")
public class ButtonApp extends Frame {

	public static void main(String[] args) {
		Frame f = new Frame();
		Button b = new Button("Beep");
		b.addActionListener(new ButtonActionListener());
		f.addWindowListener(new ButtonWindowListener());

		f.add(b);
		f.pack();
		f.setVisible(true);
		System.out.println("done");
		System.out.println("end");
	}

}

class ButtonActionListener implements ActionListener {
	public void actionPerformed(ActionEvent e) {
		System.out.println("Beep pressed");
	}
}

class ButtonWindowListener implements WindowListener {
	@Override
	public void windowOpened(WindowEvent e) { }

	@Override
	public void windowClosing(WindowEvent e) {
		System.exit(0);
	}

	@Override
	public void windowClosed(WindowEvent e) { }

	@Override
	public void windowIconified(WindowEvent e) { }

	@Override
	public void windowDeiconified(WindowEvent e) { }

	@Override
	public void windowActivated(WindowEvent e) { }

	@Override
	public void windowDeactivated(WindowEvent e) { }

}