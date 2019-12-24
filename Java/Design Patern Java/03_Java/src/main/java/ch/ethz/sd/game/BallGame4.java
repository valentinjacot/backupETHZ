package ch.ethz.sd.game;


import javax.swing.*;
import java.awt.event.*;


@SuppressWarnings("serial")
public class BallGame4 extends JFrame {
	public static void main(String[] args){
		JFrame f = new BallGame4();
		f.setDefaultCloseOperation(EXIT_ON_CLOSE);
		f.pack();
		f.setVisible(true);
	}
	
	public BallGame4(){
		setTitle("BallGame4");
		add(new BallField4());
	}
}

@SuppressWarnings("serial")
class BallField4 extends JComponent {
	// position and radius of a ball
	private int x = 50, y = 50, r = 10;
	
	public BallField4(){
		setPreferredSize(new java.awt.Dimension(300,300));
		addMouseListener(new MouseListener(this));
	}

	public void paintComponent (java.awt.Graphics g) {
		g.setColor(java.awt.Color.red);
		g.fillOval(x-r, y-r, 2*r, 2*r);
	}

	private static class MouseListener extends MouseAdapter {
		private BallField4 field;
		MouseListener(BallField4 field){
			this.field = field;
		}
		
		@Override
		public void mousePressed(MouseEvent e){
			field.x = e.getX();
			field.y = e.getY();
			field.repaint();
		}
	}


}