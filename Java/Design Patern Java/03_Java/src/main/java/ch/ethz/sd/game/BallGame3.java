package ch.ethz.sd.game;


import javax.swing.*;
import java.awt.event.*;


@SuppressWarnings("serial")
public class BallGame3 extends JFrame {
	public static void main(String[] args){
		JFrame f = new BallGame3();
		f.setDefaultCloseOperation(EXIT_ON_CLOSE);
		f.pack();
		f.setVisible(true);
	}
	
	public BallGame3(){
		setTitle("BallGame3");
		add(new BallField3());
	}
}

@SuppressWarnings("serial")
class BallField3 extends JComponent {
	// position and radius of a ball
	int x = 50, y = 50, r = 10;
	
	public BallField3(){
		setPreferredSize(new java.awt.Dimension(300,300));
		addMouseListener(new BallFieldMouseListener3(this));
	}

	public void paintComponent (java.awt.Graphics g) {
		g.setColor(java.awt.Color.red);
		g.fillOval(x-r, y-r, 2*r, 2*r);
	}
}

class BallFieldMouseListener3 extends MouseAdapter {
	private BallField3 field;
	BallFieldMouseListener3(BallField3 field){
		this.field = field;
	}
	
	@Override
	public void mousePressed(MouseEvent e){
		field.x = e.getX();
		field.y = e.getY();
		field.repaint();
	}
}
