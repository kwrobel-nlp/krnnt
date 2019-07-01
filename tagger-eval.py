#!/usr/bin/python
# -*- coding: utf-8 -*-

# Copyright (C) 2011 Adam Radziszewski.
# This program is free software; you can redistribute and/or modify it
# under the terms of the GNU Lesser General Public License as published by the Free
# Software Foundation; either version 3 of the License, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the LICENCE and COPYING files for more details
# CHANGES
# - July 2017 - added statistics for lemma accuracy for known and unknown
#               words separately (Łukasz Kobyliński)

from optparse import OptionParser
import sys
import corpus2
from StringIO import StringIO
from collections import defaultdict as dd
descr = """%prog [options] TAGDFOLD1 ... REFFOLD1 ...

Evaluates tagging of tagged corpus consisting of possibly several folds using
the given reference corpus (gold standard). The subsequent folds should match.

E.g. PYTHONIOENCODING=utf8 ./tagger-eval.py tagged??.xml folds/test??.xml | tee results.txt

"""

changelog = """
* respect_spaces
* report POS hits also
* comments
* fix debug strings
* skip non-disamb tokens as Pantera artefacts
* higher frac precision in output
* extract measures to functions for averaging
* averaging over folds
* separate stats for unknown forms
* evaluate lemmatisation lower bound (weak and strong)
* get rid of heuristic punctuation tweaks and WC measure (without bounds)
* case-insensitive lemma comparison (strong lemma nocase lower bound)
* case-sensitive and case-insensitive concatenative heuristic for upper bound
  (if concatenation of all segments in a fragment gives the same text)
* option to process only first lexeme of each token (by default off)
"""

def text(tok_seq, respect_spaces, mark_boundaries = False):
	"""Extracts text from a sequence of tokens. If respect_spaces, will append
	spaces between tokens where no no-space markers present."""
	buff = StringIO()
	nonfirst = False
	for tok in tok_seq:
		if nonfirst and respect_spaces and tok.after_space():
			buff.write(' ')
		if mark_boundaries:
			buff.write('[')
		buff.write(tok.orth_utf8().decode('utf-8'))
		if mark_boundaries:
			buff.write(']')
		nonfirst = True
	return buff.getvalue()

def next_tok(rdr):
	while True:
		tok = rdr.get_next_token()
		if not tok:
			return tok
		if tok.has_disamb_lexeme():
			return tok

def tok_seqs(rdr_here, rdr_there, respect_spaces, verbose_mode, debug_mode):
	"""Generates subsequent aligned token sequences from the two readers.
	Alignment is performed on the text level. Shortest token sequences with
	matching text are generated.
	"""
	
	tok_here = next_tok(rdr_here)
	tok_there = next_tok(rdr_there)
	
	assert tok_here and tok_there, 'no input'
	
	buff_here = [tok_here]
	buff_there = [tok_there]
	
	LIMIT = 30
	
	num_iter = 0
	
	while True:
		num_iter += 1
		if num_iter % 10000 == 0: print num_iter, 'iterations...'
		if len(buff_here) > LIMIT or len(buff_there) > LIMIT:
			raise IOError('limit exceeded')
		
		text_here = text(buff_here, respect_spaces)
		text_there = text(buff_there, respect_spaces)
		
		if debug_mode:
			print '%s (%d) v. %s(%d)' % (text_here, len(text_here),
			text_there, len(text_there))
		
		if len(text_here) == len(text_there):
			# already aligned
			if text_here != text_there:
				# the same text length but different chars
				# won't recover, flush it as it is
				print 'WARNING: mismatch', text_here, '|', text_there
			yield (buff_here, buff_there)
			buff_here = []
			buff_there = []
			tok_here = next_tok(rdr_here)
			tok_there = next_tok(rdr_there)
			if not tok_here or not tok_there:
				assert not tok_here and not tok_there, 'one ended prematurely'
				break
			buff_here = [tok_here]
			buff_there = [tok_there]
		elif len(text_here) < len(text_there):
			# try fetching here
			tok_here = next_tok(rdr_here)
			assert tok_here, 'first ended prematurely'
			buff_here.append(tok_here)
		else: # len(text_here) > len(text_there):
			# try fetching there
			tok_there = next_tok(rdr_there)
			assert tok_there, 'second ended prematurely'
			buff_there.append(tok_there)
	
	if buff_here or buff_there:
		assert buff_here and buff_there
		yield (buff_here, buff_there)

class Feat:
	WEAK_LEM_HIT = 'weak lem hit' # also includes strong hits
	STRONG_LEM_HIT = 'strong lem hit'
	STRONG_LEM_NOCASE_HIT = 'strong lem nocase hit' # case-insens as above
	WEAK_TAG_HIT = 'weak tag hit' # also includes strong hits
	STRONG_TAG_HIT = 'strong tag hit'
	WEAK_POS_HIT = 'weak pos hit' # also includes strong hits
	STRONG_POS_HIT = 'strong pos hit'
	SEG_NOCHANGE = 'segmentation unchanged'
	SEG_CHANGE = 'segmentation change'
	KNOWN = 'known'
	UNKNOWN = 'unknown'

class Metric:
	# lower bounds for lemmatisation
	WL_LOWER = ([Feat.WEAK_LEM_HIT, Feat.SEG_NOCHANGE], None) # lower bound for weak lemmatisation
	SL_LOWER = ([Feat.STRONG_LEM_HIT, Feat.SEG_NOCHANGE], None) # lower bound for strong lemmatisation
	# separate stats for known and unknown forms
	KN_SL_LOWER = ([Feat.STRONG_LEM_HIT, Feat.SEG_NOCHANGE, Feat.KNOWN], [Feat.KNOWN])
	UNK_SL_LOWER = ([Feat.STRONG_LEM_HIT, Feat.SEG_NOCHANGE, Feat.UNKNOWN], [Feat.UNKNOWN])
	SL_NOCASE_LOWER = ([Feat.STRONG_LEM_NOCASE_HIT, Feat.SEG_NOCHANGE], None) # as above but case-insensitive
	SL_NOCASE_CAT_HEUR = ([Feat.STRONG_LEM_NOCASE_HIT], None) # as above but concatenate fragments and judge if they are equal
	SL_CASE_CAT_HEUR = ([Feat.STRONG_LEM_HIT], None) # as above but case-sensitive
	
	# SL_HUPPER = ([Feat.STRONG_LEM_HIT], None) TODO: -> lemma
	# lower bounds for correctness, treating all segchanges as failures
	WC_LOWER = ([Feat.WEAK_TAG_HIT, Feat.SEG_NOCHANGE], None) # lower bound for weak correctness
	SC_LOWER = ([Feat.STRONG_TAG_HIT, Feat.SEG_NOCHANGE], None) # lower bound for strong correctness
	KN_SC_LOWER = ([Feat.STRONG_TAG_HIT, Feat.SEG_NOCHANGE, Feat.KNOWN], [Feat.KNOWN])
	UNK_SC_LOWER = ([Feat.STRONG_TAG_HIT, Feat.SEG_NOCHANGE, Feat.UNKNOWN], [Feat.UNKNOWN])
	# as above but metric for POS hits
	POS_WC_LOWER = ([Feat.WEAK_POS_HIT, Feat.SEG_NOCHANGE], None) # lower bound for POS WC
	POS_SC_LOWER = ([Feat.STRONG_POS_HIT, Feat.SEG_NOCHANGE], None) # lower bound for POS SC
	# separate stats for known and unknown forms
	KN_WC_LOWER = ([Feat.WEAK_TAG_HIT, Feat.SEG_NOCHANGE, Feat.KNOWN], [Feat.KNOWN])
	UNK_WC_LOWER = ([Feat.WEAK_TAG_HIT, Feat.SEG_NOCHANGE, Feat.UNKNOWN], [Feat.UNKNOWN])
	KN_SEG_CHANGE = ([Feat.SEG_CHANGE, Feat.KNOWN], [Feat.KNOWN])
	UNK_SEG_CHANGE = ([Feat.SEG_CHANGE, Feat.UNKNOWN], [Feat.UNKNOWN])

	KN_POS_SC_LOWER = ([Feat.STRONG_POS_HIT, Feat.SEG_NOCHANGE, Feat.KNOWN], [Feat.KNOWN])
	UNK_POS_SC_LOWER = ([Feat.STRONG_POS_HIT, Feat.SEG_NOCHANGE, Feat.UNKNOWN], [Feat.UNKNOWN])
	# percentage of known and unknown tokens
	KN = ([Feat.KNOWN], None)
	UNK = ([Feat.UNKNOWN], None)
	# percentage of tokens subjected to seg change
	SEG_CHANGE = ([Feat.SEG_CHANGE], None)
	SEG_NOCHANGE = ([Feat.SEG_NOCHANGE], None)

class TokComp:
	"""Creates a tagger evaluation comparator. The comparator reads two
	annotated texts: tagged text (output of a tagger being evaluated) and
	reference tagging (the gold standard). Most of the figures reported concern
	whether a reference token has got some sort of coverage in the tagger
	output.
	
	Differences in segmentation are treated by aligning tokens textually,
	i.e. shortest matching sequences are extracted and then compared.
	
	unk_tag is a string representation of tag used for unknown words.
	Set expand_optional to True if ommission of optional attribute
	values should be treated as multiple tags, each with a different
	variant of the value."""
	def __init__(self, tagset, unk_tag,
		expand_optional, first_lex_only, debug = False):
		self.tagset = tagset
		self.unk_tag = unk_tag
		self.expand_optional = expand_optional
		self.first_lex_only = first_lex_only
		self.debug = debug
		self.ref_toks = 0 # all tokens in ref corpus
		self.tag_toks = 0 # all tokens in tagger output
		self.tag_feat = dd(int) # feat frozenset -> count
	
	def eat_ref_toks(self, feat_set, num_toks):
		"""Classifies num_toks reference tokens as having the given set of
		features. Will also increment the ref_toks counter."""
		self.ref_toks += num_toks
		self.tag_feat[frozenset(feat_set)] += num_toks
	
	def is_unk(self, tok):
		"""There is an 'unknown word' interpretation."""
		tok_tags = [self.tagset.tag_to_string(lex.tag()) for lex in tok.lexemes()]
		return self.unk_tag in tok_tags
	
	def tagstrings_of_token(self, tok):
		"""Returns a set of strings, corresponding to disamb tags
		found in the token. If expand_optional, multiple tags may be
		produced from single-disamb tokens when some optional attributes
		are unspecified."""
		tags = [lex.tag() for lex in tok.lexemes() if lex.is_disamb()]
		assert tags
		if self.first_lex_only:
			# forcing first disamb lexeme only
			tags = tags[:1]
		if self.expand_optional:
			# create multivalue tag wherever a value of optional attr is unspec
			tags = [self.tagset.expand_optional_attrs(tag) for tag in tags]
			# now expand multivalue tags to singular tags
			newtags = []
			for tag in tags:
				newtags.extend(self.tagset.split_tag(tag))
			tags = newtags
			if self.debug:
				print ' + '.join(self.tagset.tag_to_string(tag) for tag in tags)
		return set(self.tagset.tag_to_string(tag) for tag in tags)
	
	def lemstrings_of_token(self, tok):
		"""Returns a set of unicode strings, corresponding to disamb lemmas
		found in the token."""
		if self.first_lex_only:
			# forcing first disamb lexeme only
			for lex in tok.lexemes():
				if lex.is_disamb():
					return set([unicode(lex.lemma())])
			assert False, 'no disamb lemma here'
		
		lemmas = set(
			unicode(lex.lemma())
			for lex in tok.lexemes()
			if lex.is_disamb())
		return lemmas
	
	def lem_concat(self, tok_seq, case_sens):
		"""Returns a concatenation of possibly lowercased disamb lemmas of
		tokens belonging to the given sequence. If any token has multiple
		different (lowercased) disamb lemmas, will surround them with
		parentheses and separate with |."""
		buff = StringIO()
		for tok in tok_seq:
			their_lemmas = self.lemstrings_of_token(tok)
			if case_sens:
				their_lc_lemmas = sorted(their_lemmas)
			else:
				their_lc_lemmas = sorted(
					set(lem.lower() for lem in their_lemmas))
			if len(their_lc_lemmas) > 1:
				# get u'(first|second|…)
				text = u'(%s)' % u'|'.join(their_lc_lemmas)
			else:
				text = their_lc_lemmas[0]
			buff.write(text)
		return buff.getvalue()
	
	def cmp_toks(self, tok1, tok2):
		"""Returns a set of features concerning comparison of two tokens:
		tok1 from tagger output and tok2 from ref. Actually, both should be
		interchangable, hence are labelled tok1 and tok2.
		This function is called only when token-to-token alignment is possible.
		"""
		hit_feats = set()
		tok1_lems = self.lemstrings_of_token(tok1)
		tok2_lems = self.lemstrings_of_token(tok2)
		tok1_lems_nocase = set(lem.lower() for lem in tok1_lems)
		tok2_lems_nocase = set(lem.lower() for lem in tok2_lems)
		tok1_tags = self.tagstrings_of_token(tok1)
		tok2_tags = self.tagstrings_of_token(tok2)
		tok1_pos = set(t.split(':', 1)[0] for t in tok1_tags)
		tok2_pos = set(t.split(':', 1)[0] for t in tok2_tags)
		# check lemmas in case-insensitive manner
		if tok1_lems_nocase == tok2_lems_nocase:
			hit_feats.add(Feat.STRONG_LEM_NOCASE_HIT)
		# check lemmas in case-sensitive manner
		if tok1_lems == tok2_lems:
			hit_feats.add(Feat.STRONG_LEM_HIT)
			hit_feats.add(Feat.WEAK_LEM_HIT)
		elif tok1_lems.intersection(tok2_lems):
			hit_feats.add(Feat.WEAK_LEM_HIT)
		# check POSes
		if tok1_pos == tok2_pos:
			hit_feats.add(Feat.STRONG_POS_HIT)
			hit_feats.add(Feat.WEAK_POS_HIT)
		elif tok1_pos.intersection(tok2_pos):
			hit_feats.add(Feat.WEAK_POS_HIT)
		# check full tags
		if tok1_tags == tok2_tags:
			hit_feats.add(Feat.STRONG_TAG_HIT)
			hit_feats.add(Feat.WEAK_TAG_HIT)
		elif tok1_tags.intersection(tok2_tags):
			hit_feats.add(Feat.WEAK_TAG_HIT)
		
		return hit_feats
	
	def update(self, tag_seq, ref_seq):
		self.tag_toks += len(tag_seq)
		#self.ref_toks += len(ref_seq) TODO
		
		# initialise empty feat set for each ref token
		pre_feat_sets = [set() for _ in ref_seq]
		# check if there are any "unknown words"
		for tok, feats in zip(ref_seq, pre_feat_sets):
			feats.add(Feat.UNKNOWN if self.is_unk(tok) else Feat.KNOWN)
		
		# now check for segmentation changes
		# first variant: no segmentation mess
		if len(tag_seq) == 1 and len(ref_seq) == 1:
			# there is only one token, hence one feat set
			# update it with hit feats
			pre_feat_sets[0].add(Feat.SEG_NOCHANGE)
			pre_feat_sets[0].update(self.cmp_toks(tag_seq[0], ref_seq[0]))
		else:
			if self.debug:
				print 'SEGCHANGE\t%s\t%s' % (
					text(tag_seq, True, True), text(ref_seq, True, True))
			
			# mark all as subjected to segmentation changes
			for feats in pre_feat_sets:
				feats.add(Feat.SEG_CHANGE)
			
			# now check lower-cased concatenation of ref and tag lemmas
			# if they both make up the same text (case-insensitively)
			# then we will count is is STRONG_LEM_NOCASE_HIT
			# this will be used only by the heuristic measure
			# that counts SEG_CHANGE tokens
			# NOTE: in case of lemma ambiguity there is not a simple way
			# to compare whole sequence; any lemma ambiguity will cause
			# the sequence to be unmatched
			if self.lem_concat(tag_seq, False) == self.lem_concat(ref_seq, False):
				for feats in pre_feat_sets:
					feats.add(Feat.STRONG_LEM_NOCASE_HIT)
			
			# the same with case-sensitive concatenation
			if self.lem_concat(tag_seq, True) == self.lem_concat(ref_seq, True):
				for feats in pre_feat_sets:
					feats.add(Feat.STRONG_LEM_HIT)
			#else:
				#print 'UUUU lem concat failed'
				#print 'TAG', self.lem_concat(tag_seq, True)
				#print 'REF', self.lem_concat(ref_seq, True)
			
			# check if all ref and tagged toks are punctuation
			#all_punc_ref = all(self.is_punc(tok) for tok in ref_seq)
			#all_punc_tag = all(self.is_punc(tok) for tok in tag_seq)
			#if all_punc_ref and all_punc_tag:
				#for feats in pre_feat_sets:
					#feats.update([Feat.ALLPUNC_HIT,
						#Feat.WEAK_POS_HIT, Feat.STRONG_POS_HIT,
						#Feat.WEAK_TAG_HIT, Feat.STRONG_TAG_HIT])
				## second variant: PUNC v. PUNC gives hit
				#if self.debug: print '\t\tpunc hit, ref len', len(ref_seq)
			#else:
				#nonpunc_ref = [tok for tok in ref_seq if not self.is_punc(tok)]
				#nonpunc_tag = [tok for tok in tag_seq if not self.is_punc(tok)]
				#if len(nonpunc_ref) == len(nonpunc_tag) == 1:
					## perhaps third variant: both seqs have one non-punc token
					## if the non-punc tokens match, will take the hit features
					## for the whole ref
					#hit_feats = self.cmp_toks(nonpunc_tag[0], nonpunc_ref[0])
					#for feats in pre_feat_sets:
						#feats.update(hit_feats)
					#if Feat.WEAK_TAG_HIT in hit_feats:
						#for feats in pre_feat_sets:
							#feats.add(Feat.PUNCAROUND_HIT)
						#if self.debug: print '\t\tpuncPLUS weak hit, ref len', len(ref_seq)
		# collect features assigned to each ref token in this aligned fragment
		for feats in pre_feat_sets:
			self.eat_ref_toks(feats, 1)
			if self.debug:
				print ' - ', ', '.join(sorted(feats))
	
	def count_all(self): # TODO remove
		"""Returns the number of all reference tokens."""
		return sum(self.tag_feat.values())
	
	def count_with(self, feats):
		"""Returns the number of reference tokens having all the given
		features."""
		satisfying = [key for key in self.tag_feat if key.issuperset(set(feats))]
		return sum(self.tag_feat[key] for key in satisfying)
	
	def percentage_with(self, feats, wrt_to = None):
		"""Returns the percentage of reference tokens that have the given
		features. By default all the reference tokens are treated as the
		denominator. If wrt_to given, will be used as a reference feat set."""
		count = self.count_with(feats)
		if wrt_to is not None:
			denom = self.count_with(wrt_to)
		else:
			denom = self.ref_toks
		if denom == 0:
			return 0.0 # what else can we do? should be labelled N/A
		return 100.0 * count / denom

	def value_of(self, metric):
		"""Calculates the value of the given metric, being a tuple of
		features for tokens counted as hit and features for tokens counted
		at all (or None if all the tokens should be counted). The first feats
		should be a subset of the second one."""
		return self.percentage_with(metric[0], metric[1])
	
	def dump(self):
		print '----'
		print 'REF-toks\t%d' % self.ref_toks
		for m_name in dir(Metric):
			if not m_name.startswith('_'):
				metric = getattr(Metric, m_name)
				print '%s\t%.4f%%' % (m_name, self.value_of(metric))
		# calc upper bound
		upbound = self.value_of(Metric.WC_LOWER) + self.value_of(Metric.SEG_CHANGE)
		print 'WC_UPPER\t%.4f%%' % upbound

def go():
	parser = OptionParser(usage=descr)
	parser.add_option('-i', '--input-format', type='string', action='store',
		dest='input_format', default='xces',
		help='set the input format; default: xces')
	parser.add_option('-t', '--tagset', type='string', action='store',
		dest='tagset', default='nkjp',
		help='set the tagset used in input; default: nkjp')
	parser.add_option('-q', '--quiet', action='store_false', default=True, dest='verbose')
	parser.add_option('-u', '--unk-tag', type='string', action='store',
		dest='unk_tag', default='ign',
		help='set the tag used for unknown forms; default: ign')
	parser.add_option('-k', '--keep-optional', action='store_false',
		default=True, dest='expand_optional',
		help='do not expand unspecified optional attributes to multiple tags')
	parser.add_option('-s', '--ignore-spaces', action='store_false',
		default=True, dest='respect_spaces',
		help='ignore spaces between tokens when comparing')
	parser.add_option('-f', '--first-lexeme-only', action='store_true',
		default=False, dest='first_lex_only',
		help='read only each token\'s first disamb lexeme (tag+lemma)')
	parser.add_option('-d', '--debug', action='store_true', dest='debug_mode')
	(options, args) = parser.parse_args()
	
	if len(args) < 2 or len(args) % 2 != 0:
		print 'You need to provide a series of tagged folds and a coresponding'
		print 'series of reference folds.'
		print 'See --help for details.'
		print
		sys.exit(1)
	
	tagset = corpus2.get_named_tagset(options.tagset)
	
	num_folds = len(args) / 2
	
	weak_lem_lower_bound = 0.0
	kn_strong_lem_lower_bound = 0.0
	unk_strong_lem_lower_bound = 0.0
	strong_lem_lower_bound = 0.0
	strong_lem_nocase_lower_bound = 0.0
	strong_lem_case_cat_heur = 0.0
	strong_lem_nocase_cat_heur = 0.0
	
	weak_lower_bound = 0.0
	weak_upper_bound = 0.0
	
	strong_pos_lower = 0.0

	unk_weak_lower = 0.0
	unk_weak_upper = 0.0
	kn_weak_lower = 0.0
	kn_weak_upper = 0.0
	
	perc_unk = 0.0
	perc_segchange = 0.0
	
	for fold_idx in range(num_folds):
		tag_fn = args[fold_idx] # filename of tagged fold @ fold_idx
		ref_fn = args[fold_idx + num_folds] # ... reference fold @ fold_idx
		if options.verbose:
			print '### FOLD %2d: %s (tag) v. %s (ref)' % ((fold_idx + 1), tag_fn, ref_fn)
		tag_rdr = corpus2.TokenReader.create_path_reader(options.input_format, tagset, tag_fn)
		ref_rdr = corpus2.TokenReader.create_path_reader(options.input_format, tagset, ref_fn)
		
		res = TokComp(
			tagset, options.unk_tag, options.expand_optional,
			options.first_lex_only, options.debug_mode)
		for tag_seq, ref_seq in tok_seqs(tag_rdr, ref_rdr, options.respect_spaces, options.verbose, options.debug_mode):
			res.update(tag_seq, ref_seq)

		print "PolEval 2017 competition scores"
		print "-------------------------------"
		print 'POS accuracy (Subtask A score): \t%.4f%%' % res.value_of(Metric.SC_LOWER)
		print 'POS accuracy (known words): \t%.4f%%' % res.value_of(Metric.KN_SC_LOWER)
		print 'POS accuracy (unknown words): \t%.4f%%' % res.value_of(Metric.UNK_SC_LOWER)
		print 'Lemmatization accuracy (Subtask B score): \t%.4f%%' % res.value_of(Metric.SL_LOWER)
		print 'Lemmatization accuracy (known words): \t%.4f%%' % res.value_of(Metric.KN_SL_LOWER)
		print 'Lemmatization accuracy (unknown words): \t%.4f%%' % res.value_of(Metric.UNK_SL_LOWER)
		print 'Overall accuracy (Subtask C score): \t%.4f%%' % ((res.value_of(Metric.SC_LOWER) + res.value_of(Metric.SL_LOWER)) / 2)


		if options.verbose:
			res.dump()
		
		weak_lem_lower_bound += res.value_of(Metric.WL_LOWER)
		kn_strong_lem_lower_bound += res.value_of(Metric.KN_SL_LOWER)
		unk_strong_lem_lower_bound += res.value_of(Metric.UNK_SL_LOWER)
		strong_lem_lower_bound += res.value_of(Metric.SL_LOWER)
		
		strong_lem_nocase_lower_bound += res.value_of(Metric.SL_NOCASE_LOWER)
		
		strong_lem_case_cat_heur += res.value_of(Metric.SL_CASE_CAT_HEUR)
		strong_lem_nocase_cat_heur += res.value_of(Metric.SL_NOCASE_CAT_HEUR)
		
		weak_lower_bound += res.value_of(Metric.WC_LOWER)
		weak_upper_bound += res.value_of(Metric.WC_LOWER) + res.value_of(Metric.SEG_CHANGE)
		unk_weak_lower += res.value_of(Metric.UNK_WC_LOWER)
		unk_weak_upper += res.value_of(Metric.UNK_WC_LOWER) + res.value_of(Metric.UNK_SEG_CHANGE)
		kn_weak_lower += res.value_of(Metric.KN_WC_LOWER)
		kn_weak_upper += res.value_of(Metric.KN_WC_LOWER) + res.value_of(Metric.KN_SEG_CHANGE)
		strong_pos_lower += res.value_of(Metric.POS_SC_LOWER)
		perc_unk += res.value_of(Metric.UNK)
		perc_segchange += res.value_of(Metric.SEG_CHANGE)
	
	# weak lemma -- when sets of possible lemmas output and in ref corp intersect
	print 'AVG weak lemma lower bound\t%.4f%%' % (weak_lem_lower_bound / num_folds)
	print 'AVG KN strong lemma lower bound\t%.4f%%' % (kn_strong_lem_lower_bound / num_folds)
	print 'AVG UNK strong lemma lower bound\t%.4f%%' % (unk_strong_lem_lower_bound / num_folds)
	# strong lemma -- when sets of possible lemmas output and in ref corp are equal
	print 'AVG strong lemma lower bound\t%.4f%%' % (strong_lem_lower_bound / num_folds)
	print 'AVG strong lemma nocase lower bound\t%.4f%%' % (strong_lem_nocase_lower_bound / num_folds)
	
	print 'AVG strong lemma case concat heur\t%.4f%%' % (strong_lem_case_cat_heur / num_folds)
	print 'AVG strong lemma nocase concat heur\t%.4f%%' % (strong_lem_nocase_cat_heur / num_folds)
	
	print 'AVG weak corr lower bound\t%.4f%%' % (weak_lower_bound / num_folds)
	print 'AVG weak corr upper bound\t%.4f%%' % (weak_upper_bound / num_folds)
	
	print 'AVG UNK weak corr lower bound\t%.4f%%' % (unk_weak_lower / num_folds)
	print 'AVG UNK weak corr upper bound\t%.4f%%' % (unk_weak_upper / num_folds)
	print 'AVG KN  weak corr lower bound\t%.4f%%' % (kn_weak_lower / num_folds)
	print 'AVG KN  weak corr upper bound\t%.4f%%' % (kn_weak_upper / num_folds)
	print 'AVG POS strong corr lower bound\t%.4f%%' % (strong_pos_lower / num_folds)
	
	print 'AVG percentage UNK\t%.4f%%' % (perc_unk / num_folds)
	print 'AVG percentage seg change\t%.4f%%' % (perc_segchange / num_folds)
	

if __name__ == '__main__':
	go()
