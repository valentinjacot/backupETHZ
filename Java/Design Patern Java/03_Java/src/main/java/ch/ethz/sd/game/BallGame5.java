package ch.ethz.sd.game;


import javax.swing.*;
import java.awt.event.*;


@SuppressWarnings("serial")
public class BallGame5 extends JFrame {
	public static void main(String[] args){
		JFrame f = new BallGame5();
		f.setDefaultCloseOperation(EXIT_ON_CLOSE);
		f.pack();
		f.setVisible(true);
	}
	
	public BallGame5(){
		setTitle("BallGame5");
		add(new BallField5());
	}
}

@SuppressWarnings("serial")
class BallField5 extends JComponent {
	// position and radius of a ball
	private int x = 50, y = 50, r = 10;
	
	public BallField5(){
		setPreferredSize(new java.awt.Dimension(300,300));
		addMouseListener(new MouseListener());
	}

	public void paintComponent (java.awt.Graphics g) {
		g.setColor(java.awt.Color.red);
		g.fillOval(x-r, y-r, 2*r, 2*r);
	}

	private class MouseListener extends MouseAdapter {
		@Override 
		public void mousePressed(MouseEvent e){
			x = e.getX();
			y = e.getY();
			repaint();
		}
	}

}






