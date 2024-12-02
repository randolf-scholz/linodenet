// SEE: https://github.com/mathjax/MathJax/issues/2708#issuecomment-861779454
const MathJax = {
  startup: {
    ready() {
      //
      // Do usual setup
      //
      MathJax.startup.defaultReady();
      //
      // The pseudoscript numbers 0 through 9, and a pattern for plus-or-minus a number
      //
      const scripts = '\u2070\u00B9\u00B2\u00B3\u2074\u2075\u2076\u2077\u2078\u2079';
      const scriptRE = /([\u207A\u207B])?([\u2070\u00B9\u00B2\u00B3\u2074-\u2079]+)/g;
      //
      //  Add a TeX prefilter to convert pseudoscript numbers to actual superscripts
      //
      MathJax.startup.document.inputJax[0].preFilters.add((data) => {
        data.math.math = data.math.math.replace(scriptRE, (match, pm, n) => {
          const N = n.split('').map(c => scripts.indexOf(c));  // convert digits
          pm === '\u207A' && N.unshift('+');     // add plus, if given
          pm === '\u207B' && N.unshift('-');     // add minus, if given
          return '^{' + N.join('') + '}';        // make it an actual power
        });
      });
    }
  }
};
