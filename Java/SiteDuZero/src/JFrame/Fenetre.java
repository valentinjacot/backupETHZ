package JFrame;

import javax.swing.JFrame;

public class Fenetre extends JFrame {
	  public Fenetre(){                
	    this.setTitle("Ma premi�re fen�tre Java");
	    this.setSize(100, 150);
	    this.setLocationRelativeTo(null);               
	    this.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
	    this.setContentPane(new Panneau());

	    this.setVisible(true);
	  }     
	}