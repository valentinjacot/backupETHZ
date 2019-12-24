package jdraw.action;

import java.awt.event.ActionEvent;

import javax.swing.Action;
import javax.swing.JMenu;
import javax.swing.KeyStroke;

import jdraw.framework.DrawContext;
import jdraw.framework.Figure;
import jdraw.std.SimpleClipboard;

@SuppressWarnings("serial")
public class PasteAction extends AbstractAction{
	public PasteAction(DrawContext dc, JMenu JM) {
		super(dc);
		putValue(Action.SHORT_DESCRIPTION, "Paste copied figures");
		putValue(Action.NAME, "paste");
        putValue(Action.ACCELERATOR_KEY, KeyStroke.getKeyStroke("CTRL-V"));
		JM.addMenuListener(this);
	}

	@Override
	public void actionPerformed(ActionEvent e) {
			actionContext.getView().clearSelection();
			for (Figure f : SimpleClipboard.get()) {
				f.move(10,10);
				Figure f2 = f.clone();
				actionContext.getModel().addFigure(f2);
				actionContext.getView().addToSelection(f2);
		}
	}

}
