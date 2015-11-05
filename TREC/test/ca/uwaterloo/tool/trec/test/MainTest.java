package ca.uwaterloo.tool.trec.test;

import static org.junit.Assert.*;

import org.junit.Test;

import ca.uwaterloo.tool.trec.Main;

public class MainTest {

	@Test
	public void test() {
		String[] args={"-file", "data/Relink/Apache.arff", "-labelinfo", "buggy","-source", "data","-experimental"};
		Main.main(args);
	}

}
