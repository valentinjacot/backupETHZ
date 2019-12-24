/*
 * Copyright (c) 2018 Fachhochschule Nordwestschweiz (FHNW)
 * All Rights Reserved. 
 */

package jdraw.commands;

import java.util.Stack;

import jdraw.framework.DrawCommand;
import jdraw.framework.DrawCommandHandler;

/**
 * Provides an empty command handler. This class provides an empty dummy implementation of the draw command
 * handler. It enables the application to start up correctly and to behave meaningful, but with the limitation
 * that it does not provide any undo/redo behavior. 
 * @author Christoph. Denzler
 *
 */
public class ListDrawCommandHandler implements DrawCommandHandler {
	private Stack<DrawCommand> undoStack;
	private Stack<DrawCommand> redoStack;
	private Script script;
	
	
	public ListDrawCommandHandler() {
		super();
		this.undoStack = new Stack<DrawCommand>();
		this.redoStack = new Stack<DrawCommand>();
	}

	@Override
	public void addCommand(DrawCommand cmd) {
		redoStack.clear();
		if (script == null){
			undoStack.push(cmd);	
		}else {
			script.addCommand(cmd);
		}
	}
	
	@Override
	public void undo() {
		if (undoPossible()) {
			DrawCommand cmd = undoStack.pop();
			redoStack.push(cmd);
			cmd.undo();
		}
	}

	@Override
	public void redo() { 
		if (redoPossible()) {
			DrawCommand cmd = redoStack.pop();
			undoStack.push(cmd);
			cmd.redo();
		}
	}

	@Override
	public boolean undoPossible() { return !undoStack.empty(); }

	@Override
	public boolean redoPossible() { return !redoStack.empty(); }

	@Override
	public void beginScript() {
		if (script != null) throw new IllegalStateException();
		script = new Script(); 
	}

	@Override
	public void endScript() { 
		if (script == null) throw new IllegalStateException();
		Script tmp = script;
		script = null;
		if(tmp.commands.size() > 0)
			addCommand(tmp);
	}

	@Override
	public void clearHistory() { 
		undoStack.clear();
		redoStack.clear();
	}
}
