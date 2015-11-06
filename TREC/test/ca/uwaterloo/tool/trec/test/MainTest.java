package ca.uwaterloo.tool.trec.test;

import static org.junit.Assert.*;

import org.junit.Test;

import ca.uwaterloo.tool.trec.Main;

public class MainTest {

	@Test
	public void test() {
		String[] args={"-file", "data/Relink/Apache.arff", "-labelinfo", "buggy","-source", "data","-experimental"};
		Main.main(args);
		args[1] = "data/Relink/Safe.arff";
		Main.main(args);
		
		args[1] = "data/Relink/Zxing.arff";
		Main.main(args);
		
		args[3] = "TRUE";
		args[1] = "data/AEEEM/EQ.arff";
		Main.main(args);
		
		args[1] = "data/AEEEM/JDT.arff";
		Main.main(args);
		
		args[1] = "data/AEEEM/LC.arff";
		Main.main(args);
		
		args[1] = "data/AEEEM/ML.arff";
		Main.main(args);
		
		args[1] = "data/AEEEM/PDE.arff";
		Main.main(args);
	}
}
