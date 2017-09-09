//#!/bin/sh
//exec scala "$0" "$@"
//!#

import scala.io.Source
import scala.io.BufferedSource
import scala.util.control.Breaks

object W2VSkipGram {
  
  def main(args: Array[String]): Unit = {

    val window_size = 1
    val input_file = "/Users/rem/data/text8.txt"
    
    val simpleTokenizer = new Iterator[String]() {
      val s:BufferedSource = Source.fromFile(input_file)
      val next_string:StringBuilder = new StringBuilder
      def next() : String = {
        val next = next_string.toString
        next_string.clear
        return next;
      }
      def hasNext(): Boolean = {
        Breaks.breakable {
          while(s.hasNext) {
            val next_char = s.next
            if(next_char == ' ' || next_char == '\n')
              Breaks.break
            next_string.append(next_char)
          }
        }
        val hasnext = !next_string.isEmpty
        if(!hasnext)
          s.close()
        return hasnext
      }
    }
    
    simpleTokenizer.toIterable.sliding(window_size + window_size + 1).foreach { x =>
      println(String.format("%s\t%s", x.slice(window_size, window_size+1).mkString, (x.slice(0, window_size) ++ x.slice(window_size+1, x.size)).mkString(" ") )) 
    }
    
  }
  
}
