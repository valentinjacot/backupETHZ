package patterns.state.tickets;

import java.awt.GridLayout;
import java.awt.event.ActionEvent;

import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JFrame;
import javax.swing.JTextField;

@SuppressWarnings("serial")
public class TicketMachineGUI extends JFrame {
	private final JTextField destination = new JTextField();
	private final JTextField price = new JTextField();
	private final JTextField money = new JTextField();
	private final JCheckBox halfPrice = new JCheckBox("1/2");
	private final JCheckBox firstClass = new JCheckBox("1st");
	private final JCheckBox retour = new JCheckBox("<=>");
	private final JButton cancel = new JButton("Cancel");
	
	private final TicketMachine t = new TicketMachine2();
	
	TicketMachineGUI() {
		this.setLayout(new GridLayout(0,1));
		this.add(destination);
		this.add(price); price.setEditable(false);
		this.add(halfPrice);
		this.add(firstClass);
		this.add(retour);
		this.add(money);
		this.add(cancel);
		
		destination.addActionListener(e -> {
				t.setDestination(Integer.parseInt(destination.getText()));
				update(t, t.getPrice());
			}
		);
		
		halfPrice.addActionListener(e -> {
				t.setHalfPrice(halfPrice.isSelected());
				update(t, t.getPrice());
			}
		);
		
		firstClass.addActionListener(e -> {
				t.setFirstClass(firstClass.isSelected());
				update(t, t.getPrice());
			}
		);
		
		retour.addActionListener(e -> {
				t.setReturnTicket(retour.isSelected());
				update(t, t.getPrice());
			}
		);
		
		money.addActionListener((ActionEvent e) -> {
				t.enterMoney(Double.parseDouble(money.getText()));
				money.setText("");
				update(t, t.getPrice()-t.getEnteredMoney());
			}
		);
	
		cancel.addActionListener(e -> {
				t.cancel();
				update(t, 0.0);
			}
		);
	
		update(t, 0);
		pack();
		setDefaultCloseOperation(EXIT_ON_CLOSE);
	}

	private void update(TicketMachine t, double p) {
		price.setText(String.format("%4.2f", Math.max(0,  p)));
		halfPrice.setSelected(t.isHalfPrice());
		firstClass.setSelected(t.isFirstClass());
		retour.setSelected(t.isRetour());
		
		if(t.isInStateInit()) destination.setText("");
		
		destination.setEnabled(t.isInStateInit());
		money.setEnabled(t.isInStateDestSelected() || t.isInStateMoneyEntered());
		halfPrice.setEnabled(t.isInStateDestSelected());
		firstClass.setEnabled(t.isInStateDestSelected());
		retour.setEnabled(t.isInStateDestSelected());
	}

	public static void main(String[] args) {
		JFrame f = new TicketMachineGUI();
		f.setVisible(true);
	}

}
