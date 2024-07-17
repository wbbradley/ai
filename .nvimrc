augroup sql
  autocmd FileType python setlocal makeprg=./check
augroup END
nmap <Leader>` :!ctags -R . .venv/lib<CR>
